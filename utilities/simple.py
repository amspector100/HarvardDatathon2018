"""Contains (i) nonspatial helper functions, (ii) simple spatial processing functions, and
    (iii) a couple of shapely-based manipulations"""

import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

# Misc --------------------------------------------------------------
def will_it_float(text):
    """Checks whether an object can be converted to a float."""
    try:
        text = float(text)
        return True
    except ValueError:
        return False

def convert_to_hex(rgba_color):
    """ Converts rgba colors to hexcodes. Adapted from
            https://stackoverflow.com/questions/35516318/plot-colored-polygons-with-geodataframe-in-folium
    """
    red = str(hex(int(rgba_color[0]*255)))[2:].capitalize()
    green = str(hex(int(rgba_color[1]*255)))[2:].capitalize()
    blue = str(hex(int(rgba_color[2]*255)))[2:].capitalize()

    if blue=='0':
        blue = '00'
    if red=='0':
        red = '00'
    if green=='0':
        green='00'

    return '#'+ red + green + blue


# Processing functions ----------------------------------------------------------------------------------------

def retrieve_coords(point):
    """Retrieves coords and reverses their order for shapely point. (Reverses because folium and GeoPandas use opposite lat/long conventions)."""
    result = list(point.coords[:][0][0:])
    result.reverse()
    return result

def process_geometry(gdf, geometry_column = 'geometry', drop_multipolygons = True):
    """Processing for polygon-based gdfs: makes geometries valid and possibly drops multipolygons."""

    gdf = gdf.loc[gdf[geometry_column].apply(lambda x: x is not None)]
    gdf.loc[:, geometry_column] = gdf[geometry_column].apply(lambda poly: poly if poly.is_valid else poly.buffer(0))

    # Drop multipolygons and warn user if this is a bad idea
    if drop_multipolygons:
        gdf = gdf.loc[gdf[geometry_column].apply(lambda x: isinstance(x, shapely.geometry.polygon.Polygon))]
        if gdf.shape[0] == 0:
            warnings.warn('In process_geometry call, dropping polygons may have eliminated all the data')

    return gdf

def process_points(points, geometry_column = 'geometry'):
    """Processing for point-based gdfs: ignores invalid points"""
    points = points.loc[(points[geometry_column].is_valid) & (points[geometry_column].apply(lambda x: x is not None))]
    points.reset_index(drop=True, inplace=True)
    points.index = [str(ind) for ind in points.index]
    return points

# Part 2: Spatial utilities -------------------------------------------------------------------------------------------

def fragment(polygon, horiz = 10, vert = 10):
    """

    Fragment polygon into smaller pieces. This is used to (vastly) improve spatial tree efficiency on large polygons.

    :param polygon: Polygon to fragment
    :type polygon: shapely polygon
    :param horiz: Number of horizontal fragments, defaults to 10
    :type horiz: int
    :param vert: Number of vertical fragments, defaults to 10
    :type vert: int
    :return: A list of smaller polygons which are a partition of the input polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    xlen = (maxx - minx) / horiz
    ylen = (maxy - miny) / vert
    grid = []
    for i in np.arange(0, horiz, 1):
        for j in np.arange(0, vert, 1):
            b = shapely.geometry.box(xlen * i + minx, ylen * j + miny, xlen * (i + 1) + minx,
                    ylen * (j + 1) + miny)  # Left, botton, right, upper
            g = polygon.intersection(b)
            if g.is_empty:
                continue
            grid.append(g)

    return grid

# Get urban cores (i.e. circle of certain radius around a lat long)
def get_urban_core(lat, long, radius, scale = 5280, newproj = 'epsg:2277'):
    """

    Create a polygon representing the urban core of a city. Based on the shapely's buffer method, but combined with
    crs transformations so you get to pick the units.

    :param lat: The latitude of the center of the city.
    :param long: The longitude of the center of the city.
    :param radius: The radius in units of your choice; see the scale and newproj parameters.
    :param scale: Defaults to 5280, feet per mile.
    :param newproj: The new projection to use to calculate this distance (by default epsg:2277, which is in feet).
    :return: a geopandas geodataframe with a single column (geometry) of length one (polygon) which represents the
        urban core.

    """

    # Create point, transform to new coords. Do long lat because there's no standardization.
    core = gpd.GeoDataFrame(geometry = [shapely.geometry.point.Point(long, lat)])
    core.crs = {'init':'epsg:4326'}
    core = core.to_crs({'init':newproj})

    # Get point and create buffer
    core = core['geometry'][0]
    core = core.buffer(scale*radius)
    core = gpd.GeoDataFrame(geometry = [core])
    core.crs = {'init':newproj}

    # Return to dataframe and transform back to lat long
    core = core.to_crs({'init':'epsg:4326'})
    return core

def make_point_grid(gdf, horiz=20, vert=20, factor=None, by='mean', geometry_column='geometry'):
    """

    Given a geodataframe of points, partition them into a rectangular grid and calculate either the number of points in
    each rectangle or the mean or median of a factor associated with the points for each rectangle. This is used to
    make choropleths out of point data.

    :param gdf: Geodataframe, with point geometry presumably.
    :param horiz: Number of horizontal boxes
    :param vert: Number of vertical boxes
    :param factor: The (continuous) value with which to take the mean/median of the points. Defaults to None.
    :param by: 'mean' or 'median'. Meaningless unless you have the factor column.
    :param geometry_column: The column the points are contained in (these should be shapely points).
    :return: geodataframe with grid geometry and a 'value' column
    """

    # Work with crs
    if gdf.crs is None:
        warnings.warn('Assuming crs is lat/long')
        gdf.crs = {'init': 'epsg:4326'}

    # Fragment
    points = shapely.geometry.multipoint.MultiPoint(gdf[geometry_column].tolist())
    hull = points.convex_hull
    grid = fragment(hull, horiz = horiz, vert = vert)

    # Now initialize result
    result = gpd.GeoSeries(grid)
    result.index = [str(ind) for ind in result.index]  # prevent weirds errors arising from spatial index
    value = pd.Series(index=result.index)

    # Get intersections and fill values
    spatial_index = gdf.sindex
    for ind in result.index:
        box = result[ind]
        possible_matches_index = list(spatial_index.intersection(box.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches.loc[possible_matches[geometry_column].intersects(box)]

        if factor is None:
            value[ind] = len(precise_matches)
        elif by == 'mean':
            value[ind] = precise_matches.loc[precise_matches[factor].notnull(), factor].mean()
        elif by == 'median':
            value[ind] = precise_matches.loc[precise_matches[factor].notnull(), factor].median()
        else:
            raise ValueError('In make_point_grid call, "by" argument must either equal "mean" or "median" - you put "{}"'.format(by))

    result = gpd.GeoDataFrame(data = pd.DataFrame(data = value.values, columns = ['value'], index = value.index), geometry = result)
    result.crs = gdf.crs
    return result
