import pandas as pd
import matplotlib


file = pd.read_csv("data/food_establishment_inspections.csv", engine='python')
print(file.head())