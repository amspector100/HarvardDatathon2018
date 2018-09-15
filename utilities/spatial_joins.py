"""Fast intersection functions"""

import pandas as pd
import geopandas as gpd
import warnings
from . import simple

def points_intersect_single_polygon(points, polygon, spatial_index, points_geometry_column = 'geometry',
                             factors = None, categorical = True, by = 'mean', **kwargs):
    """

    Given many points and a polygon, finds one of three things. (1) If factors = None, the number of points inside the
    polygon, (2) if factors is not None and categorical = True, the number of points inside the polygon conditional on
    a group of categorical factors, (3) if factors is not None and categorical = False, the summarized value (mean/median)
    of factors associated with each point of each point inside the polygon.

    :param points: A GDF with a points geometry column
    :param polygon: The polygon to see whether the points are inside.
    :param spatial_index: The spatial index of the points
    :param factors: The factors to average over (if continuous) or subset by the cartesian product of (if categorical).
        This may be a list or a string.
    :param categorical: If True, then the factor should be treated as a categorical variable.
    :param by: If categorical is False, can either summarize using by = 'mean' or by = 'median'
    :param kwargs: Kwargs to pass to the "fragment" function in the TXHousing.utilities.simple module. Fragmenting polygons
        speeds up the computation for all but very small polygons. If you do not want to fragment the polygons (the
        only reason to do this is speed, it will not affect the results), pass in horiz = 1 and vert = 1 as kwargs.
    :return: Pandas series

    Note: it is often useful to apply this function to an entire gdf of polygons.

    """

    # Get intersections
    all_precise_matches_indexes = set()
    grid = simple.fragment(polygon, **kwargs)
    for grid_piece in grid:
        possible_matches_index = list(spatial_index.intersection(grid_piece.bounds))
        possible_matches = points.iloc[possible_matches_index]

        # Prevent weird indexing errors by ignoring empty results
        if possible_matches.shape[0] == 0:
            continue

        precise_matches_index = set(possible_matches.index[possible_matches.intersects(grid_piece)].tolist())
        all_precise_matches_indexes = all_precise_matches_indexes.union(precise_matches_index)

    if factors is None:
        return len(precise_matches_index)
    else:
        precise_matches = points.loc[all_precise_matches_indexes]
        if categorical == True:
            return precise_matches.groupby(factors)[points_geometry_column].count()
        elif by == 'mean':
            return precise_matches[factors].mean()
        elif by == 'median':
            return precise_matches[factors].median()
        else:
            raise ValueError(
            'In points_intersect_polygon call, "by" must either equal "mean" or "median," not "{}"'.format(by))


def polygons_intersect_single_polygon(small_polygons, polygon, spatial_index, geometry_column = 'geometry',
                                      factors = None, categorical = True, account_for_area = True,
                                      divide_area_by = 'polygon', by = 'mean', **kwargs):
    """

    Given many polygons (i.e. parcels) and a larger polygon (i.e. county boundary), finds one of three things.
    (1) If factor = None, the percent area of the large polygon that is covered by the small polygons
    (2) If factor is not None and categorical = True, the percent area of the large polygon that is covered by the small
    polygons conditional on the factor
    (3) if factor is not None and categorical = False, the summarized value (mean/median) of the factors associated with
    each polygon inside the polygon.

    :param small_polygons: A GDF with a polygon geometry column
    :param polygon: The polygon to see whether the small_polygons are inside.
    :param spatial_index: The spatial index of the small_polygons
    :param factors: The factors to average over (if continuous) or subset by the cartesian product of (if categorical).
    :param categorical: If True, factors will be treated as categorical variables.
    :param by: If categorical is False, can summarize with by = 'mean' or by = 'median'
    :param account_for_area: Default True. If True, instead of returning the mean of the factor, this will return the
        dot product of the mean and the area of each small_polygon that intersects the large_polygon divided by the area
        of the large polygon (happens if categorical is False, by = 'mean', and account_for_area = True). Also,
        if factor = None, divides answer by area of polygon.
    :param divide_area_by: Defaults to 'polygon'. This parameter determines what to divide the result by.
        If divide_area_by = 'polygon', then this divides by the area of the polygon. If divide_area_by = 'nonempty', it
        will divide by the total area of the intersection between the polygon and small_polygons. Else, it will simply
        return without dividing.
    :param kwargs: Kwargs to pass to the "fragment" function in the TXHousing.utilities.simple module. Fragmenting polygons
        speeds up the computation for all but very small polygons. If you do not want to fragment the polygons (the
        only reason to do this is speed, it will not affect the results), pass in horiz = 1 and vert = 1 as kwargs.
    :return: Float if factors is None, else Pandas Series

    Note: it is often useful to apply this function to an entire gdf of polygons.

    """

    # Get the indexes of the small polygons which intersect the big polygon
    precise_matches_indexes = set()
    grid = simple.fragment(polygon, **kwargs)
    for grid_piece in grid:
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = small_polygons.iloc[possible_matches_index]

        # Might speed things up a tiny bit; also might prevent weird indexing errors
        if possible_matches.shape[0] == 0:
            continue

        precise_matches_to_add = set(possible_matches.index[possible_matches.intersects(grid_piece)].tolist())
        precise_matches_indexes = precise_matches_indexes.union(precise_matches_to_add)

    # Calculate actual intersections
    precise_matches = small_polygons.loc[precise_matches_indexes]
    precise_matches.loc[:, geometry_column] = precise_matches.loc[:, geometry_column].intersection(polygon)
    precise_matches['area'] = precise_matches[geometry_column].area

    # Calculate answers
    if factors is None:
        if account_for_area and divide_area_by == 'polygon':
            return precise_matches['area'].sum()/polygon.area
        elif divide_area_by == 'nonempty':
            warnings.warn('Cannot divide_area_by by the total area of the nonempty intersections when factor is None')
            return precise_matches['area'].sum()/polygon.area
        else:
            return precise_matches['area'].sum()

    # Group by categorical
    elif categorical == True:
        if account_for_area and divide_area_by == 'nonempty':
            return precise_matches.groupby(factors)['area'].sum()/(precise_matches['area'].sum())
        elif account_for_area and divide_area_by == 'polygon':
            return precise_matches.groupby(factors)['area'].sum()/(polygon.area)
        else:
            return precise_matches.groupby(factors)['area'].sum()
    else:
        factor_data = precise_matches[factors]

        # Transpose if there are multiple factors (for dot product)
        if isinstance(factors, str) == False:
            factor_data = factor_data.T

        if by == 'mean':

            # Different division methods
            if account_for_area and divide_area_by == 'nonempty':
                return factor_data.dot(precise_matches['area'])/(precise_matches['area'].sum())
            elif account_for_area and divide_area_by == 'polygon':
                return factor_data.dot(precise_matches['area'])/(polygon.area)
            elif account_for_area:
                return factor_data.dot(precise_matches['area'])
            else:
                return factor_data.mean()

        elif by == 'median':
            return factor_data.median()
        else:
            raise ValueError(
            'In points_intersect_polygon call, "by" must either equal "mean" or "median," not "{}"'.format(by))


def points_intersect_multiple_polygons(points_gdf, polygons_gdf, points_spatial_index = None,
                              points_geometry_column = 'geometry', polygons_geometry_column = 'geometry',
                              polygons_names_column = None, **kwargs):
    """

    Given a gdf of points and a gdf of polygons, calculates the polygon in which each point lies. This function assumes
    that each point will lie in at most one of the polygons. If that assumption is not true, use instead the
    points_intersect_single_polygon function and apply it to the geometry column of a polygon gdf.

    :param points_gdf: A geodataframe of points data.
    :param polygons_gdf: A geodataframe of polygons data.
    :param points_spatial_index: Optional; the spatial_index of the points geodataframe. If not supplied, the function
        will automatically generate the spatial index.
    :param points_geometry_column: Geometry column for the points data.
    :param polygons_geometry_column: Geometry column for the polygon data.
    :param polygons_names_column: Column for the names of each polygon; if none will use the index of the polygons_gdf.
    :param kwargs: Kwargs to pass to the "fragment" function in the TXHousing.utilities.simple module. Fragmenting polygons
        speeds up the computation for all but very small polygons. If you do not want to fragment the polygons (the
        only reason to do this is speed, it will not affect the results), pass in horiz = 1 and vert = 1 as kwargs.
    :return: A pandas series mapping the index of the points_gdf to the names of the polygons. If an index does not
        appear in the returned series, that is because the point corresponding to that index did not lie inside any of
        the polygons.
    """

    result = {}
    if points_spatial_index is None:
        points_spatial_index = points_gdf.sindex

    # Create lists of polygons/names from large_polygon_gdf
    polygon_list = polygons_gdf[polygons_geometry_column].values.tolist()
    if polygons_names_column is not None:
        names_list = polygons_gdf[polygons_names_column].values.tolist()
    else:
        names_list = polygons_gdf.index.tolist()

    # Loop through and find intersections
    warning_count = 0
    for name, polygon in zip(names_list, polygon_list):

        # Fragment the polygons.
        grid = simple.fragment(polygon, **kwargs)
        for grid_piece in grid:
            possible_intersections_index = list(points_spatial_index.intersection(grid_piece.bounds))
            possible_intersections = points_gdf.iloc[possible_intersections_index]
            precise_intersections_bools = possible_intersections[points_geometry_column].intersects(grid_piece)
            precise_intersections = possible_intersections[precise_intersections_bools].index.tolist()
            for i in precise_intersections:
                if i in result and result[i] != name:
                    warning_count += 1
                result[i] = name

    # Warn the user if a point lies in multiple polygons
    if warning_count != 0:
        warnings.warn("""In points_intersect_polygons, up to {} points are in multiple polygons, but points_intersect_polygons
         only returns one polygon per point (if a point is in two polygons, it will only show up as being in one).""".format(warning_count))

    result = pd.Series(result)
    return result


def fast_polygon_intersection(small_polygon_gdf, large_polygon_gdf, small_points_spatial_index = None,
                              small_geometry_column = 'geometry', large_geometry_column = 'geometry',
                              large_name_column = None, **kwargs):
    """

    Given a gdf of small polygons (i.e. parcels) and a gdf of large polygons (i.e. municipal boundaries), calculates the
    large polygon in which each small polygon lies. This function is based on the points_intersect_multiple_polygons
    function and therefore assumes that each small polygon will lie in at most one of the large polygons.

    :param small_polygon_gdf: A gdf of small polygons (i.e. parcels)
    :param large_polygon_gdf: A gdf of large polygons (i.e. municipal boundaries)
    :param small_points_spatial_index: The spatial index for the centroids of the small_polygon_gdf. Note that passing
        the spatial index for the polygons of the small_polygon_gdf is different and could lead to unexpected results.
        This is optional, and will be generated by the underling points_intersect_multiple_polygons call if not supplied.
    :param small_geometry_column: The geometry column of the small_polygon_gdf.
    :param large_geometry_column: The geometry column of the large_polygon_gdf.
    :param large_name_column: Column for the names of each large polygon; if none will use the index of the large_polygon_gdf.
    :param kwargs: Kwargs to pass to the "fragment" function in the TXHousing.utilities.simple module. Fragmenting polygons
        speeds up the computation for all but very small polygons. If you do not want to fragment the polygons (the
        only reason to do this is speed, it will not affect the results), pass in horiz = 1 and vert = 1 as kwargs.
    :return: A pandas series mapping the index of the small polygons to the names of the large polygons. If an index
        does not appear in the returned series, that is because the small polygon corresponding to that index did not
        lie inside any of the large polygons.
    """

    # Get centroids
    if 'centroids' not in small_polygon_gdf.columns:
        small_polygon_gdf['centroids'] = small_polygon_gdf[small_geometry_column].centroid

    small_polygon_gdf = small_polygon_gdf.set_geometry('centroids')

    result = points_intersect_multiple_polygons(points_gdf = small_polygon_gdf, polygons_gdf = large_polygon_gdf,
                                                points_spatial_index = small_points_spatial_index,
                                                points_geometry_column = 'centroids',
                                                polygons_geometry_column = large_geometry_column,
                                                polygons_names_column = large_name_column, **kwargs)

    small_polygon_gdf.set_geometry(small_geometry_column) # Undo global effects on small_polygon_gdf

    return result


def get_averages_by_area(data_source, other_geometries, features, density_flag = False, data_source_geometry_column = 'geometry',
                             other_geometries_column = 'geometry', drop_multipolygons = True, account_method = None,
                        horiz = 1, vert = 1):
    """

    Get averages of features from data_source by area. Data_source and other_geometries should have the
    same crs initially. This is a wrapper for polygons_intersect_single_polygon and is therefore quite accurate.

    :param data_source: The data source, usually block data. Must have polygon geometry.
    :type data_source: GeoDataFrame
    :param other_geometries:  Will calculate features each row of this gdf from the data source. Must have polygon
        geometry.
    :param features: The feature in question. Can also be a list of features, i.e. ['B01001e1', 'B01001e2']
    :type features: str or list
    :param density_flag: Default False. If True, will assume that the 'feature' is already units per area and will not
        divide the feature by the area of the data source polygons.
    :type density_flag: Boolean
    :param data_source_geometry_column: geometry column for data_source
    :param other_geometries_column: geometry column for other_geometries
    :param account_method: The method by which to account for the % of an area which is not residential (this prevents
        population-related estimates from being too low). Can either be None, 'percent_residential', or 'percent_land'.
        Defaults to None (although wrappers of this function may have different defaults).
    :param horiz: When fragmenting polygons, number of horizontal fragments to make. Defaults to 1.
    :param vert: When fragmenting polygons, number of vertical fragments to make. Defaults to 1.
    :return: other_geometries but with a new column, feature, which has the averages by area.
    """

    # Process features and make sure we won't be overwriting anything
    if isinstance(features, str):
        features = [features]
    overwritten_features = [feature for feature in features if feature in other_geometries.columns]

    if len(overwritten_features) != 0:
        raise AttributeError("""Some features are already present in other_geometries and would be overwritten by 
        get_averages_by_area: they are {}.""".format(overwritten_features))

    # Process crs
    if data_source.crs != other_geometries.crs:
        raise AttributeError("""The crs for data_source ({}) and other_geometries ({}) disagree.""".format(data_source.crs, other_geometries.crs))

    # Process data for convenience (just to prevent multipolygons/invalid polygons from messing things up)
    data_source = simple.process_geometry(data_source, drop_multipolygons=drop_multipolygons)
    data_source.reset_index(drop = True)
    data_source.index = [str(ind) for ind in data_source.index]

    # Start to rename the features to prevent global side effects
    old_columns_dictionary = {str(feature) + '_density':feature for feature in features}
    new_columns_dictionary = {feature:str(feature) + '_density' for feature in features}
    new_columns = [new_columns_dictionary[key] for key in new_columns_dictionary]

    if density_flag == False:
        # If the features are not per area (i.e. population), then divide them to get feature densities (ie. population density)

        # Account for the percent of land in block groups that may not be populated. Default to None. This is mostly useful
        # for calculating populations of very small regions, i.e. population of municipal zones.

        if account_method == 'percent_residential':
            # Using percent_residential accounting method
            data_source = data_source.loc[data_source['percent_residential'] >= 0.01] # Ignore blocks which are < .5% resid
            densities = data_source[features].divide(data_source['percent_residential'], axis = 0)
            densities = densities.divide(data_source[data_source_geometry_column].area, axis = 0)
        elif account_method == 'water':
            # Using water accounting method
            data_source.loc[:, 'percent_land'] = data_source['ALAND'].divide(data_source["ALAND"] + data_source['AWATER'])
            densities = data_source[features].multiply(data_source['percent_land'], axis = 0)
            densities = densities.divide(data_source[data_source_geometry_column].area, axis = 0)
        else:
            # Not using any accounting method
            densities = data_source[features].divide(data_source[data_source_geometry_column].area, axis = 0)

    else:
        # If the features are already in units per area, then don't divide them.
        densities = data_source[features].copy()

    densities = densities.rename(columns = new_columns_dictionary)
    data_source = data_source.join(densities)

    other_geometries = simple.process_geometry(other_geometries, drop_multipolygons = drop_multipolygons)

    # Get spatial index
    data_spatial_index = data_source.sindex

    # Quick function to apply to geometry column for other_geometries
    def get_avg(polygon):
        result = polygons_intersect_single_polygon(data_source, polygon, data_spatial_index, factors = new_columns,
                                                   categorical = False, account_for_area = True, divide_area_by = None,
                                                   by = 'mean', horiz = horiz, vert = vert)
        return result

    # Get averages by area - this takes a while.
    final_values = other_geometries[other_geometries_column].apply(get_avg)

    # If the features were provided as densities, return them as densities
    if density_flag:
        final_values = final_values.divide(other_geometries[other_geometries_column].area, axis = 0)

    # Rename columns
    final_values = final_values.rename(columns = old_columns_dictionary)

    # Join and return
    if isinstance(final_values, pd.Series):
        assert len(features) == 1
        final_values.name = features[0]

    other_geometries = other_geometries.join(final_values)

    return other_geometries