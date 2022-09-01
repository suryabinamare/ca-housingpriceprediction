from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pickle

# load the housing data. 
housing = pd.read_csv('housing.csv')
print(housing.head())
X = housing[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]
y = housing['MedHouseVal']
model2 = LinearRegression()
model2.fit(X,y)
# make pickle file of ur model
pickle.dump(model2,open('model2.pkl','wb'))

