"""Functions which focus on efficienctly measuring distances and area."""

import scipy
import shapely
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import geopandas as gpd
import warnings
from . import simple, spatial_joins

texas_places_path = "data/Census/cb_2017_48_place_500k/cb_2017_48_place_500k.shp"


# Haversine, area, dist to center - None of these use spatial joins -------------------------------------------------

def haversine(point1, point2, lon1 = None, lat1 = None, lon2 = None, lat2 = None):
    """

    Haversine function calculates distance (in miles) between two points in lat/long coordinates. See
    https://gis.stackexchange.com/questions/279109/calculate-distance-between-a-coordinate-and-a-county-in-geopandas

    :param point1: Shapely point. Long then lat.
    :param point2: Shapely point. Long then lat.
    :param lon1, lat1, lon2, lat2: Alternatively, supply the longitudes and lattitudes manually.
    :return: Distance in miles.
    """

    # Retrieve lat and long
    if lon1 is None or lat1 is None:
        lon1, lat1 = list(point1.coords[:][0][0:])
    if lon2 is None or lat2 is None:
        lon2, lat2 = list(point2.coords[:][0][0:])

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(scipy.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = scipy.sin(dlat / 2) ** 2 + scipy.cos(lat1) * scipy.cos(lat2) * scipy.sin(dlon / 2) ** 2
    c = 2 * scipy.arcsin(scipy.sqrt(a))
    r = 3956  # Radius of earth in miles. Use 6371 for km
    return c * r

def calculate_dist_to_center(gdf, lat, long, drop_centroids = True, geometry_column = 'geometry'):
    """

    Calculates distance to the center of the city using haversine on the centroids of objects

    :param gdf: A GeoDataFrame, with lat/long crs. Can either have point or polygon geometry.
    :param lat: Latitude of the center of the city
    :param long: Longitude of the center of the city.
    :param drop_centroids: Boolean, default true. If true, drop the centroids inplace after calculation.
    :return: Pandas Series of floats (distances from center).
    """

    if gdf.crs != {'init':'epsg:4326'}:
        raise AttributeError('gdf must be in lat/long coordinates for calculate_dist_to_center')

    if 'centroids' not in gdf.columns:
        gdf['centroids'] = gdf['geometry'].centroid

    # Get lats and longs for gdf and center respectively
    lat = lat * np.ones((gdf.shape[0]))
    long = long * np.ones((gdf.shape[0]))
    gdf['long'] = gdf['centroids'].apply(lambda x: x.coords[:][0][0])
    gdf['lat'] = gdf['centroids'].apply(lambda x: x.coords[:][0][1])

    distances = haversine(point1 = None, point2 = None,
                          lat1 = lat, lon1 = long,
                          lat2 = gdf['lat'], lon2 = gdf['long'])
    if drop_centroids:
        gdf.drop(['centroids', 'lat', 'long'], inplace=True, axis=1)
    return distances

def get_area_in_units(gdf, geometry_column = 'geometry', newproj = 'epsg:2277', scale = 3.58701*10**(-8), name = 'area', final_projection = None, reproject = True):
    """

    Get the area of each polygon of a geodataframe in units of your choice (defaults to square miles). This function
    relies on crs transformations, so for large/complex gdfs, this function is very computationally expensive.

    :param gdf: Geodataframe with polygons in the geometry column.
    :param geometry_column: Geometry column of the geodataframe, defaults to 'geometry'
    :param newproj: The new projection to use to calculate units. Defaults to epsg:2277, which is probably fine for
        Austin/Dallas/Houston and is in feet.
    :param scale: A scale to multiply by. Defaults to 3.58701*10**(-8) which is the number of square miles in a square
        foot.
    :param name: The name of the new column that will be created to store the area information. Defaults to 'area'.
    :param final_projection: The final projection that the returned gdf should be in. Defaults to the gdf's current crs.
    :param reproject: If False, do not reproject the data after calculating area (this is useful to save time in specific
        cases).
    :type reproject: Boolean

    :return: The geodataframe with a column named name (defaults to 'area') which has the area of each polygon in
    the desired units.

    """

    # Get final projection
    if final_projection is None:
        final_projection = gdf.crs

    # Project and get area
    gdf = gdf.to_crs({'init':newproj})
    gdf[name] = scale*gdf[geometry_column].area

    # Optionally reproject (usually do this)
    if reproject:
        gdf = gdf.to_crs(final_projection)

    return gdf

# These functions group gdfs into successive rings in distance from city center. They employ spatial joins and are
# somewhat expensive.


# Part 5: Create dist from city center graphs, for points and polygons ------------------------------------------------

def order_radii(data, inplace = True, feature = None ):
    """Helper function which properly orders wacky indexes/features for pandas dataframes. If inplace = False, works
    with a copy of the data to prevent global effects. """

    # Sort all the floatable values (i.e. 1, 2.5, 3, 4, 5).
    if inplace == False:
        data = data.copy()

    if feature is not None:
        data.set_index(feature, inplace = True)

    values = data.index.unique()

    # Consider some floats
    sorted_values = sorted([float(v) for v in values if simple.will_it_float(v)])

    # Need to process index at the same time to avoid NaNs from cropping up
    def process_index(item):
        if simple.will_it_float(item):
            item = float(item)
            if item.is_integer():
                item = int(item)
        item = str(item)
        return item
    data.index = data.index.map(process_index)
    sorted_values = list(map(process_index, sorted_values))

    # Add the other values to the final list
    other_values = [str(v) for v in values if not simple.will_it_float(v)]
    if len(other_values) > 1:
        warnings.warn('order_radii may not order properly if more than one value cannot be coerced to a float')

    sorted_values.extend(other_values)

    # Create categorical dtype
    radii =  CategoricalDtype(sorted_values, ordered = True)
    data.index = data.index.astype(radii)

    return data


def points_intersect_rings(gdf, lat, long, factor = None, step = 1, categorical = True, by = 'mean',
                           geometry_column = 'geometry', per_square_mile = True, maximum = None):
    """
    Given a gdf of points, calculates the distance of each point from the center of the city. Can also group by
    a categorical variable or alternatively calculate the mean/median of a continuous variable.

    :param gdf: GDF in points geometry.
    :param lat: The latitude of the center of the rings.
    :param long: The longitude of the center of the rings.
    :param factor: A factor of the gdf to condition on or calculate means/medians of, e.g. 'Race' or'Population'
    :param step: Number of miles where the ring radiates outwards.
    :param maximum: Max radius (miles)
    :param categorical: If true, will only calculate percent land (or % of points) in the radius, conditional on the
        factor if factor is not None.
    :param by: Defaults to "mean". If categorical = False, use "by" to determine how to calculate averages over points.
    :param geometry_column: name of geometry column. Default geometry.
    :param per_square_mile: if true, divide by the area of the ring.
    :param maximum: float, defualts to None. If not None, will group everything greater than this maximum into a single
        category.
    :return: If factor is None, a pd Series which lists the number of points by distance from city center. If
        categorical = True and factor is None, a pandas Dataframe which lists the number of points by distance  from the
        city center (index) against their categorical value from the factor (columns). Lastly, if categorical = False and
        factor is not None, returns a pd Series of the mean/median of the factor conditional on distance to city center.
    """

    # Get distance from center of city, in miles currently
    center = shapely.geometry.point.Point(long, lat)

    def rounded_dist_to_center(point):
        dist = haversine(point, center)
        rdist = step*((dist // step) + 1) # Do this to get smoother bins (i.e. always round up)
        return rdist

    # Handle crs and apply
    if gdf.crs != {'init':'epsg:4326'}:
        warnings.warn('In points_intersect_rings call, forced to transform gdf to lat long')
        gdf = gdf.to_crs({'init':'epsg:4326'})

    gdf['dist_to_center'] = gdf[geometry_column].apply(rounded_dist_to_center)

    # Group ouotliers together if told
    if maximum is not None:
        def group_outliers(dist_to_center):
            if dist_to_center > maximum:
                return str(maximum) + '+'
            else:
                return dist_to_center
        gdf['dist_to_center'] = gdf['dist_to_center'].apply(group_outliers)
        if per_square_mile:
            warnings.warn('In points_intersect_rings, dropping all points about the maximum because per_square_mile is True.')
            gdf = gdf.loc[gdf['dist_to_center'].apply(simple.will_it_float)]


    # Get counts if no factor is provided
    if factor is None:
        result = gdf[['geometry', 'dist_to_center']].groupby('dist_to_center')['geometry'].count()

    # Get counts by factor level if factor is categorical
    elif categorical:
        result = gdf[['dist_to_center', factor]].groupby(['dist_to_center', factor]).size()
        result = result.unstack().fillna(0)

    # Get mean and median else
    elif by == 'mean':
        result = gdf[[factor, 'dist_to_center']].groupby('dist_to_center')[factor].mean()
    elif by == 'median':
        result = gdf[[factor, 'dist_to_center']].groupby('dist_to_center')[factor].median()

    # Warn user if they supplied a bad arg
    else:
        raise TypeError("""In points_intersect_rings call, when categorical = False and factors is not None, "by" argument must
        either equal "mean" or "median" - you put "{}" """.format(by))

    # If told, get area per square mile
    if per_square_mile:

        def get_area_ring(radius):
            return scipy.pi*(radius**2 - (radius - step)**2)

        areas = pd.Series(result.index.map(get_area_ring).tolist(), index = result.index)

        if isinstance(result, pd.DataFrame):
            result = result.divide(areas, axis = 0)
        else:
            result = result.divide(areas)

        # Return result
        result = order_radii(result)

    return result


# Radius from city center for polygon data
def polygons_intersect_rings(gdf, lat, long, factor = None, newproj = 'epsg:2277', step = 1,
                             maximum = 20, categorical = True, geometry_column = 'geometry',
                             group_outliers = True, outlier_maximum = 35, city = None):

    """

    Given a gdf of polygons, groups the polygons by distance from the center of the city and calculates the
    percent of area of the city that the polygons cover. Can also group by a categorical variable or alternatively
    calculate the mean/median of a continuous variable (adjusting for the area of the polygons).

    :param gdf: Geopandas GeoDataFrame, in polygon geometry.
    :param factor: A factor of the gdf to condition on or calculate means/medians of e.g. 'Race' or 'Population'
    :param lat: The latitude of the center of the rings.
    :param long: The longitude of the center of the rings.
    :param newproj: the new projection system necessarily used in this. Defaults to 2277 which is good for Austin and
        fine for Texas. Note units in this are in feet.
    :param step: Number of miles where the ring radiates outwards.
    :param maximum: Max radius (miles)
    :param geometry_column: name of geometry column. Default geometry.
    :param categorical: If true, will only calculate percent land (or % of points) in the radius. Else will calculate
        mean by area.
    :param city: If city is notnot "none", if factor is "none", will read the shapefile of the boundaries of the city
        to ensure more accurate calculations. (Otherwise, for a ring of size 12, the area of the circle might be
        greater than the area of the city inside the circle).
    :param group_outliers: Boolean, defaults to true. If true, group everything with a distance greater than the maximum
        into one group (of maximum size).
    :param outlier_maximum: Float, defaults to 35. For computational efficiency, this function will not consider outliers
        higher than this distance from the cneter of the city.
    :return: Dataframe or Series

    Note::

    To ensure accurate results, this function will break up polygons which straddle the boundary between being
    (for example) 5-6 miles from the city center as opposed to 6-7 miles from the city center; this makes it
    computationally expensive. To calculate similar results more cheaply with slightly less accuracy, just take the
    centroids of the geodataframe and apply points_intersect_rings.
    """

    feet_to_mile = 5280
    gdf = simple.process_geometry(gdf, geometry_column = geometry_column)
    gdf.reset_index(drop = True)
    gdf.index = [str(ind) for ind in gdf.index]

    # Necessarily will have to transform into new epsg code in order to efficiently calculate this.
    # Here, gdf should be in (lat, long) format
    if gdf.crs is None:
        warnings.warn('No CRS set, assuming lat and long data before transformation.')
        gdf.crs = {'init':'epsg:4326'}

    gdf = gdf.to_crs({'init':newproj})


    # Get center and initialize output
    center_latlong = shapely.geometry.point.Point(long, lat)
    center_series = gpd.GeoSeries(center_latlong)
    center_series.crs = {'init':'epsg:4326'}
    center_gdf = gpd.GeoDataFrame(geometry = center_series)
    center_gdf = center_gdf.to_crs({'init':newproj})
    center = center_gdf.loc[0, 'geometry']

    # Possibly get city shape
    if city is not None:
        place_shapes = gpd.read_file(texas_places_path)
        place_shapes.crs = {'init': 'epsg:4326'}
        city_shape = place_shapes.loc[place_shapes['NAME'] == city, 'geometry'].values[0]

    # Initialize result. If categorical, need a dataframe (one column for each unique value). Else, use a pd.Series.
    if factor is None or categorical == False:
        result = pd.Series()
    else:
        result = pd.DataFrame(columns = gdf[factor].unique().tolist())

    if categorical == False and factor is not None:

        # To make this play nicely with regulation data, just in case. Can't harm either way.
        gdf = gdf.loc[gdf[factor].notnull()]

    gdf.reset_index(drop = True, inplace = True)
    gdf.index = [str(ind) for ind in gdf.index]
    spatial_index = gdf.sindex


    radius = 0
    while radius <= maximum - step:

        radius += step
        circle = center.buffer(feet_to_mile*radius).difference(center.buffer(feet_to_mile*(radius - step)))
        if city is not None:
            circle = circle.intersection(city_shape)

        # Get kwargss to pass to fragment call (we will fragment more as the circle we're dealing with gets bigger)
        horiz = np.ceil(np.sqrt(radius))
        vert = np.ceil(np.sqrt(radius))

        # Call intersection function
        result.loc[radius] = spatial_joins.polygons_intersect_single_polygon(gdf, circle, spatial_index,
                                                                             factors = factor, categorical = categorical,
                                                                             account_for_area = True, divide_area_by = 'nonempty',
                                                                             by = 'mean', horiz = horiz, vert = vert)

        # And adjust to sum to 1 for categorical calls
        if categorical and factor is not None:
            result.loc[radius] = result.loc[radius]/(result.loc[radius].sum())


    # If this optional arg is true, then lump everything else into one category
    if group_outliers:

        # Create circle
        circle = center.buffer(feet_to_mile*outlier_maximum).difference(center.buffer(feet_to_mile*radius))

        # In this case, the denominator is the total area within the ring
        label = str(radius) + '+'

        result.loc[label] = spatial_joins.polygons_intersect_single_polygon(gdf, circle, spatial_index,
                                                                             factors = factor, categorical = categorical,
                                                                             account_for_area = True, divide_area_by = 'nonempty',
                                                                             by = 'mean', horiz = horiz, vert = vert)

        result = order_radii(result)

    return result
