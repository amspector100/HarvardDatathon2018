"""Functions that help create the mastermaps and other folium maps"""
import numpy as np

# Import graphing related helpers
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster, HeatMap, HeatMapWithTime
import branca.colormap as cm

# Import other submodules
from .BindColorMap import BindColormap

# To do
# 1. Add legend functionality for categorical choropleth, somehow.


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

def retrieve_coords(point):
    """Retrieves coords and reverses their order for shapely point. (Reverses because folium and GeoPandas use opposite lat/long conventions)."""
    result = list(point.coords[:][0][0:])
    result.reverse()
    return result

# To do
# 1. Add legend functionality for categorical choropleth, somehow.


# Marker Cluster processing function
def make_marker_cluster(gdf, make_centroids = True, points_column = 'geometry', fast = False, name = None, show = False,
                        basemap = None, **kwargs):
    """
    Makes a marker cluster from a gdf and potentially adds it to a map/feature group.

    :param GeoDataFrame gdf: A geodataframe.
    :param bool make_centroids: If true and the geodataframe has polygon geometry, it will make the centroids and use those
        to make the marker cluster.
    :param str or int points_column: If make_centroids is False, will assume the pointa are in this column.
        Defaults to 'geometry'.
    :param bool fast: If True, use a FastMarkerCluster as opposed to a regular MarkerCluster.
    :param str name: Defaults to None. If not None, will generate a FeatureGroup with this name and return that instead of
        the MarkerCluster object.
    :param bool show: Defaults to False. The show parameter for the FeatureGroup that the marker cluster will be added
        to.
    :param folium.Map basemap: Defaults to None. If not none, will add the MarkerCluster or FeatureGroup to the supplied basemap.
    :param kwargs: kwargs to pass to the FastMarkerCluster or MarkerCluster initialization
    :return: Either a FeatureGroup or a MarkerCluster.
    """


    # Possibly make centroids
    if make_centroids:
        gdf['centroids'] = gdf['geometry'].centroid
        points_column = 'centroids'

    # Retrieve points and make markercluster
    points = [retrieve_coords(point) for point in gdf[points_column]]

    if fast:
        marker = FastMarkerCluster(points, **kwargs)
    else:
        marker = MarkerCluster(points, **kwargs)

    # Possibly create featuregroup and possibly add to basemap; then return
    if name is not None:
        feature_group = folium.FeatureGroup(name, show = show)
        marker.add_to(feature_group)
        if basemap is not None:
            feature_group.add_to(basemap)
        return feature_group
    else:
        if basemap is not None:
            marker.add_to(basemap)
        return marker

def polygon_layer(gdf, color = 'blue', weight = 1, alpha = 0.6, name = None, show = False, basemap = None):
    """

    :param gdf:
    :param factor:
    :param color:
    :param weight:
    :param alpha:
    :param str name: Defaults to None. If not None, will generate a FeatureGroup with this name and return that instead of
        the GeoJson object.
    :param bool show: Defaults to False. The show parameter for the FeatureGroup that the GeoJson will be added to.
    :param folium.Map basemap: Defaults to None. If not none, will add the GeoJson or FeatureGroup to the supplied basemap.
    :return:
    """

    gjson = folium.GeoJson(
                gdf,
                style_function=lambda feature: {
                    'fillColor': color,
                    'color': color,
                    'weight': weight,
                    'fillOpacity': alpha,
                }
            )

    if name is not None:
        feature_group = folium.FeatureGroup(name, show = show)
        gjson.add_to(feature_group)
        if basemap is not None:
            feature_group.add_to(basemap)
        return feature_group
    else:
        if basemap is not None:
            gjson.add_to(basemap)
        return gjson


def categorical_choropleth(gdf, factor, colors = None, quietly = False, weight = 1, alpha = 0.6,
                           geometry_column = 'geometry', name = None, show = False, basemap = None):
    """
    Creates categorical choropleth using tab10 spectrum


    :param gdf: A geopandas geodataframe.
    :param factor: The feature you want to plot (should be categorical).
    :param colors: Colors to use in the categorical plot. If None, will generate colors using the tab10 colormap.
    :param quietly: If true, will not print anything. Defaults to False.
    :param weight: The weight in the style function. Defaults to 1.
    :param alpha: The alpha in the style function. Defaults to 0.6.
    :param geometry_column: The geometry column of the gdf. Defaults to 'geometry'.
    :param str name: Defaults to None. If not None, will generate a FeatureGroup with this name and return that instead of
        the GeoJson object.
    :param bool show: Defaults to False. The show parameter for the FeatureGroup that the GeoJson will be added to.
    :param folium.Map basemap: Defaults to None. If not none, will add the GeoJson or FeatureGroup to the supplied basemap.
    :return: A folium geojson or featuregroup.
    """

    values = gdf[factor].unique()

    # Get colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(values)))
        colors = [convert_to_hex(color) for color in colors]
    elif len(colors) < len(values):
        raise IndexError('In categorical_choropleth call, the "colors" input has fewer colors than the data has unique values')

    # Get colordic and apply to data
    colordic = {value: color for value, color in zip(values, colors)}
    gdf['color'] = gdf[factor].map(colordic)
    if not quietly:
        print('Legend functionality is not available in categorical choropleths yet, so instead we print the colordic')
        print('Here it is: {}'.format(colordic))

    # Transform data, as always
    gdf = gdf.to_crs({'init': 'epsg:4326'})

    gdf = gdf[[factor, geometry_column, 'color']]

    gjson = folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                'fillColor': feature['properties']['color'],
                'color': feature['properties']['color'],
                'weight': weight,
                'fillOpacity': alpha,
            }
        )

    # Possibly create featuregroup and possibly add to basemap; then return
    if name is not None:
        feature_group = folium.FeatureGroup(name, show = show)
        gjson.add_to(feature_group)
        if basemap is not None:
            feature_group.add_to(basemap)
        return feature_group
    else:
        if basemap is not None:
            gjson.add_to(basemap)
        return gjson

def continuous_choropleth(gdf, factor, layer_name, scale_name = None, weight = 1, alpha = 0.6,
                          colors = ['blue', 'green', 'yellow', 'orange', 'red'],
                          quants = [1/6, 2/6, 3/6, 4/6, 5/6],
                          method = 'log', round_method = None,
                          show = False, geometry_column = 'geometry', basemap = None):
    """
    :param gdf: Geodataframe
    :param factor: factor for analysis
    :param layer_name: Name of feature group layer
    :param scale_name: Name of scale
    :param weight: Weight
    :param alpha: Alpha of polygons
    :param colors: A list of colors to use in the colormap, defaults to ['blue', 'green', 'yellow', 'orange', 'red'].
    :param quants: The quantiles to use to 'switch' colors in the colormap. Defaults to [1/6, 2/6, 3/6, 4/6, 5/6].
        If you want a log-based or linear colorscale, adjust the 'method' parameter and set quants to None.
    :param method: The method by which the color scale is generated. Defaults to 'log', can also be 'quant' or 'linear'.
        This parameter is overridden by the "quantiles" parameter.
    :param round_method: If you want to round the color scale to integer values, supply round_method = 'int'
    :param show: Show by default on start
    :param geometry_column: 'geometry'
    :param basemap: If not None, will add the colormap and a scale (bound together) to the baesmap as a layer.
    :return: GeoJson, Colormap
    """

    # Get rid of nas
    gdf = gdf.loc[(gdf[factor].notnull()) & (gdf[geometry_column].notnull())]

    # Create colormap with caption
    min_data = gdf[factor].min()
    max_data = gdf[factor].max()
    if quants is not None:
        index = gdf[factor].quantile(quants)
        if len(colors) != len(index):
            raise IndexError('index and colors must be same length')
        colormap =  cm.LinearColormap(colors = colors, vmin = min_data, vmax = max_data, index = index)
    else:
        colormap =  cm.LinearColormap(colors = colors, vmin = min_data, vmax = max_data).to_step(12, method = method, round_method = round_method)


    if scale_name is None:
        colormap.caption = layer_name
    else:
        colormap.caption = scale_name

    # Create gjson
    gdf = gdf[[factor, geometry_column]]
    gjson = folium.GeoJson(
            gdf,
            show = show,
            name = layer_name,
            style_function = lambda feature: {
                'fillColor': colormap(feature['properties'][factor]),
                'color': colormap(feature['properties'][factor]),
                'weight': weight,
                'alpha': alpha,
            }
        )

    # This is for backwards compatability but always do this, it saves time
    if basemap is not None:
        colormap.add_to(basemap)
        gjson.add_to(basemap)
        BindColormap(gjson, colormap).add_to(basemap)

    return gjson, colormap


def heatmap(gdf, geometry_column = 'geometry', with_time = False, time_column = 'Year', name = None, show = False, basemap = None, **kwargs):
    """
    Create a heatmap or a heatmap with time from a geodataframe of points.

    :param gdf: Geodataframe with points as the geometry type.
    :param geometry_column: The geometry column of the gdf. Defaults to 'geometry'
    :param start_color: The start color, defaults to 'white'
    :param end_color: The end color, defaults to the MI blue
    :param with_time: If true, plot a heat map with time, not just a heat map.
    :param time_column: The column used to specify the years of the data, defaults to 'Year'
    :param str name: Defaults to None. If not None, will generate a FeatureGroup with this name and return that instead of
        the GeoJson object.
    :param bool show: Defaults to False. The show parameter for the FeatureGroup that the GeoJson will be added to.
    :param folium.Map basemap: Defaults to None. If not none, will add the GeoJson or FeatureGroup to the supplied basemap.
    :param **kwargs: kwargs to be passed onto the 'heatmap' or 'heatmapwithtime' folium constructors.
    :return: HeatMap object or FeatureGroup
    """

    if with_time:
        all_points = []
        time_periods = sorted(gdf[time_column].unique().tolist())
        for time_period in time_periods:
            points = gdf.loc[gdf[time_column] == time_period, geometry_column]
            points = [retrieve_coords(point) for point in points]
            all_points.append(points)
        result = HeatMapWithTime(all_points, index = time_periods, **kwargs)

    else:
        points = [retrieve_coords(point) for point in gdf[geometry_column]]
        result = HeatMap(points, **kwargs)


    if name is not None:
        feature_group = folium.FeatureGroup(name, show = show)
        result.add_to(feature_group)
        if basemap is not None:
            feature_group.add_to(basemap)
        return feature_group
    else:
        if basemap is not None:
            result.add_to(basemap)
        return result