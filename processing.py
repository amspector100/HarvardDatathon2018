import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry 
import matplotlib.pyplot as plt 

# Import utilities package
import utilities 
import time 

# Paths
requests_path = 'data/official/311_service_requests.csv'
inspections_path = 'data/official/food_establishment_inspections.csv'
venues_path = 'data/official/food_venues.csv'
nta_boundaries_path = 'data/NTA_Boundaries/geo_export_352cba9b-c010-4110-a340-153f171d1098.shp'

def lat_long_to_gdf(data, latitude_key = 'latitude', longitude_key = 'longitude'):
	""" Turns a pandas dataframe into a geodataframe based on lat/long coordinates"""

	# Get rid of entries without geocoding
	data = data.loc[(data[latitude_key].apply(lambda x: x is not None)) & (data[longitude_key].apply(lambda x: x is not None))]
	data = data.loc[(data[latitude_key].notnull()) & (data[longitude_key].notnull())]

	# Turn to gdf
	def make_shapely_point(row):
		lat = row[latitude_key]
		lon = row[longitude_key]
		return shapely.geometry.point.Point([lon, lat])

	data = gpd.GeoDataFrame(data, geometry = data.apply(make_shapely_point,axis = 1))
	data.crs = {'init':'epsg:4326'}
	data = data.loc[data['geometry'].is_valid]

	return data 



# Start by exploring 311 dataset -----------------------------------
def process_requests_data(path = requests_path, test = False): 
	"""
	Takes path of original 311 dataset as an input. Cleans and returns the following variables:

	1. Response time (resolution_date - created_date)
	2. Status ()
	3. 
	"""

	print("Processing requests data - this might take a minute")
	requests_data = pd.read_csv(requests_path)
	if test:
		requests_data = requests_data.iloc[0:100]
		print(requests_data.shape)

	# Subset to useful statuses
	requests_data = requests_data.loc[requests_data['status'].isin(['Assigned', 'Open', 'Closed', 'Pending', 'Started'])]

	# Add response_time variable
	requests_data['created_date'] = pd.to_datetime(requests_data['created_date'], format='%m/%d/%Y %H:%M:%S %p')
	requests_data['resolution_date'] = pd.to_datetime(requests_data['resolution_date'], format='%m/%d/%Y %H:%M:%S %p')
	requests_data['response_time'] = requests_data['resolution_date'] - requests_data['created_date']

	# Turn into gdf
	requests_data = lat_long_to_gdf(requests_data)

	# Add nta location
	nta_boundaries = gpd.read_file(nta_boundaries_path)
	requests_data['nta_key'] = utilities.spatial_joins.points_intersect_multiple_polygons(requests_data, nta_boundaries, 
																						  points_spatial_index=None, points_geometry_column='geometry', 
																					 	  polygons_geometry_column='geometry', polygons_names_column=None, 
																					 	  horiz = 1, vert = 1)
	requests_data['ntacode'] = requests_data['nta_key'].map(nta_boundaries['ntacode'])

	return requests_data

requests_data = process_requests_data(test = True)
