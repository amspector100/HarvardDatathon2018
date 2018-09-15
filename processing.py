import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry 
import datetime as dt
import matplotlib.pyplot as plt 

# Import utilities package
import utilities 
import time 

# Paths
requests_path = 'data/official/311_service_requests.csv'
inspections_path = 'data/official/food_establishment_inspections.csv'
venues_path = 'data/official/food_venues.csv'
nta_boundaries_path = 'data/NTA_Boundaries/geo_export_352cba9b-c010-4110-a340-153f171d1098.shp'
county_boundaries_path = 'data/NYS_Civil_Boundaries/Counties.shp'


# Helper functions ------------------------------------------------------------------------------------------------------------

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

def pull_boundaries(point_data, boundary_path, boundary_name, horiz = 1, vert = 1, **kwargs):
	""" Adds a column to point data which has the name of the boundary """

	boundaries = gpd.read_file(boundary_path)
	boundaries = boundaries.to_crs({'init':'epsg:4326'})
	boundaries['geometry'] = boundaries['geometry'].simplify(tolerance = 0.0005)
	point_data[boundary_name] = utilities.spatial_joins.points_intersect_multiple_polygons(point_data, boundaries, horiz = horiz, vert = vert, **kwargs)
	point_data[boundary_name] = point_data[boundary_name].map(boundaries[boundary_name])

	return point_data

def pull_counties_and_ntas(point_data, time0, pull_counties = True):
	""" Pulls counties and ntas point data. A wrapper for pull_boundaries function"""

	# Turn into gdf
	print("Parsing geometry at time {}".format(time.time() - time0))
	point_data = lat_long_to_gdf(point_data)

	# Add nta/county location
	point_data = pull_boundaries(point_data, nta_boundaries_path, 'ntacode', horiz = 2, vert = 2)
	print('Finished with ntacodes at time {}'.format(time.time() - time0))

	if pull_counties:
		point_data = pull_boundaries(point_data, county_boundaries_path, 'NAME', horiz = 4, vert = 4)
		print('Finished with counties at time {}'.format(time.time() - time0))

	return point_data


# Start with 311 dataset ----------------------------------------------------------
def process_requests_data(path = requests_path, test = False, output_path = 'data/cleaned/cleaned_311.csv'): 
	"""
	Takes path of original 311 dataset as an input. Cleans and returns the following variables (as well as others in the dataset): 

	1. resolution_time (resolution_date - created_date)
	2. response_time (closed_date - created_data)
	3. Status
	4. ntacode
	5. county
	"""

	print("Processing requests data - this might take a minute")
	time0 = time.time()
	if test:
		requests_data = pd.read_csv(requests_path, nrows = 100000)
	else:
		requests_data = pd.read_csv(requests_path)

	# Subset to useful statuses -- 
	requests_data = requests_data.loc[requests_data['status'].isin(['Assigned', 'Open', 'Closed', 'Pending', 'Started'])]

	# Add response_time variable -----
	print("Parsing request times at time {}".format(time.time() - time0))
	requests_data['created_date'] = pd.to_datetime(requests_data['created_date'], format='%m/%d/%Y %H:%M:%S %p')
	requests_data['closed_date'] = pd.to_datetime(requests_data['closed_date'], format='%m/%d/%Y %H:%M:%S %p')
	requests_data['resolution_date'] = pd.to_datetime(requests_data['resolution_date'], format='%m/%d/%Y %H:%M:%S %p')

	# Put NaNs in place of bad resolution/closed dates. Also, ignore cases where closed/resolution date is BEFORE created date. 
	requests_data.loc[(requests_data['resolution_date'] > dt.date(2018, 9, 10)), 'resolution_date'] = float("NaN")
	requests_data.loc[requests_data.apply(lambda row: row['resolution_date'] < row['created_date'], axis = 1), 'resolution_date'] = float("NaN")

	requests_data.loc[(requests_data['closed_date'] > dt.date(2018, 9, 10)), 'closed_date'] = float("NaN")
	requests_data.loc[requests_data.apply(lambda row: row['closed_date'] < row['created_date'], axis = 1), 'closed_date'] = float("NaN")

	# Add response/resolution time
	requests_data['response_time'] = (requests_data['closed_date'] - requests_data['created_date']).apply(lambda x: x.days) # NaN for Open
	requests_data['resolution_time'] = (requests_data['resolution_date'] - requests_data['created_date']).apply(lambda x: x.days) # NaN for Open

	# Clean interactions between status and response/resolution
	requests_data.loc[requests_data['closed_date'].notnull(), 'Status'] = 'Closed' # If it's closed, it's closed

	# Pull counties and nta
	requests_data = pull_counties_and_ntas(requests_data, time0)

	if output_path is not None:
		save_data = requests_data[[col for col in requests_data.columns if col != 'geometry']]
		save_data.to_csv(output_path)

	return requests_data


if __name__ == '__main__':

	process_inspection_data(test = True)
	#process_requests_data(test = False)