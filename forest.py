
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("train.csv")
#this function will provide the descriptive statistics of the dataset.(only int value)
dataset.describe()


#determine X and y variables(this values are taken as independent variables and dependent variable)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y , test_size=0.2, random_state = 0, shuffle = True )

#Random forset regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,y_train)

#predict the test data
y_pred = regressor.predict(X_test)

#r2score
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)