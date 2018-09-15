import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry 
import datetime as dt
import matplotlib.pyplot as plt 

from sklearn.ensemble import *
from sklearn.linear_model import *

np.random.seed(110)

# Demographic data path
dem_path = 'shared_data/nta_data.csv'

dem_features = ['pop', 'below_pov', 'median_hh_income', 'edu_college']
complaint_types = ['Water System', 'Dirty Conditions', 'Sanitation Condition', 'Rodent',
				 'Air Quality', 'Indoor Air Quality', 'Hazardous Materials', 'Asbestos', 
				 'Drinking', 'Water Quality', 'Mold']

def merge_and_preprocessing(nrows, depvar):
	request_data = pd.read_csv('data/cleaned/cleaned_311.csv', nrows = nrows)
	request_data = request_data.loc[request_data['ntacode'].notnull()]
	request_data = request_data.loc[request_data['complaint_type'].isin(complaint_types)]
	request_data = request_data[[depvar, 'ntacode']]

	# Get and normalize dem_data
	dem_data = pd.read_csv('shared_data/nta_data.csv', index_col = 'nta_code')
	dem_data = dem_data.astype(np.float64)
	dem_mean = dem_data.mean()
	dem_std = dem_data.std()
	dem_data = (dem_data - dem_mean)/dem_std

	# Merge data
	request_data = request_data.merge(dem_data, how = 'inner', left_on = 'ntacode', right_on = 'nta_code')

	return request_data

def model_times(test = False, dependent_variable = 'response_time'):
	
	if test:
		nrows = 100000
	else:
		nrows = None 

	request_data = merge_and_preprocessing(nrows, depvar = dependent_variable)

	# Exclude outliers
	maximum = request_data[dependent_variable].quantile(0.95)
	request_data.loc[request_data[dependent_variable] > maximum, dependent_variable] = maximum
	request_data = request_data.loc[request_data[dependent_variable].notnull()]

	X = request_data[[col for col in request_data.columns if col not in [dependent_variable, 'ntacode']]].values
	Y = request_data[dependent_variable].values

	state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(state)
	np.random.shuffle(Y)

	cutoff = int(4*X.shape[0]/5)
	X_train = X[0:cutoff]
	Y_train = Y[0:cutoff]
	X_test = X[cutoff:-1]
	Y_test = Y[cutoff:-1]

	# Random forest
	model = Ridge()
	model.fit(X_train, Y_train)
	predictions = model.predict(X_test)
	print(sum((predictions - Y_test)**2))

	print(model.score(X_test, Y_test))



if __name__ == '__main__':

	model_times(test = False)

