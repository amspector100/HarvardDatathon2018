import pandas as pd


def get_inspection_count(fpath):
	data = pd.read_csv(fpath, engine='python')
	nyc_zip_set = {10001.0, 10002.0, 10003.0, 10004.0, 10005.0, 10006.0, 10007.0, 10009.0, 10010.0, 10011.0, 10012.0, 10013.0, 10014.0, 10016.0, 10017.0, 10018.0, 10019.0, 10020.0, 10021.0, 10022.0, 10023.0, 10024.0, 10025.0, 10026.0, 10027.0, 10028.0, 10029.0, 10030.0, 10031.0, 10032.0, 10033.0, 10034.0, 10035.0, 10036.0, 10037.0, 10038.0, 10039.0, 10040.0, 10044.0, 10065.0, 10069.0, 10075.0, 10103.0, 10110.0, 10111.0, 10112.0, 10115.0, 10119.0, 10128.0, 10153.0, 10154.0, 10165.0, 10167.0, 10168.0, 10169.0, 10170.0, 10171.0, 10172.0, 10173.0, 10174.0, 10177.0, 10271.0, 10279.0, 10280.0, 10282.0}
	cleaned_data = data.loc[data["ZIPCODE"].isin(nyc_zip_set)]