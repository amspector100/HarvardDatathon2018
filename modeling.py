import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry 
import datetime as dt
import matplotlib.pyplot as plt 

def model_response_time(test = False):
	
	if test:
		nrows = 1000
	else:
		nrows = None 

	request_data = pd.read_csv('data/cleaned/cleaned_311.csv', nrows = nrows)
	request_data = request_data.loc[request_data['ntacode'].notnull()]

	nta_data = pd.read_csv('shared_data/nta_data.csv')
	print(nta_data.columns)
	print(nta_data.index)


if __name__ == '__main__':
	model_response_time(test = True)

