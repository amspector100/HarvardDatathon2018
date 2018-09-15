import pandas as pd

def get_inspection_count(fpath):
	data = pd.read_csv(fpath, engine='python')
	cleaned_data = data.drop_duplicates(['facility', 'address', 'inspection_date'])
	return cleaned_data