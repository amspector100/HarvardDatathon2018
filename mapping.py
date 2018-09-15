import pandas as pd
import geopandas as gpd
import folium 
import matplotlib.pyplot as plt
from folext import choropleth

nta_boundaries_path = 'data/NTA_Boundaries/geo_export_352cba9b-c010-4110-a340-153f171d1098.shp'
nyc_lat = 40.7128
nyc_long = -74.006

def plot_nta_boundaries(save_path = 'plots/interactive.html'):

	# Read in data and simplify (very complex) boundaries a little bit to reduce file size
	boundary_data = gpd.read_file(nta_boundaries_path)
	boundary_data['geometry'] = boundary_data['geometry'].simplify(tolerance = 0.0005)

	basemap = folium.Map([nyc_lat, nyc_long],
                     zoom_start=11)	

	choropleth.categorical_choropleth(boundary_data, factor = 'boroname', name = 'Boroughs',
									quietly = True, basemap = basemap)

	folium.TileLayer('cartodbdark_matter').add_to(basemap)
	folium.LayerControl().add_to(basemap)

	basemap.save(save_path)

if __name__ == '__main__':
	plot_nta_boundaries()