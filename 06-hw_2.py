# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.api as sm

dataset = pd.read_csv("datasets/maaslar_yeni.csv")

print(dataset)

#%%Data Preprocessing
print(dataset.corr())
#print(dataset.corr()["maas"].sort_values())
#sns.countplot(y=dataset["unvan"])

X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 5:].values

#%% 1__Create Model(linear regression)
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X, y)

print("Simple Linear OLS")
model = sm.OLS(lin_regr.predict(X), X)

print(model.fit().summary())

#%% 2a__Create Model(polynomial regression)
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)

print(X_poly)

poly_regr = LinearRegression()
poly_regr.fit(X_poly, y)

print("Polynomial OLS")
model2 = sm.OLS(poly_regr.predict(poly_features.fit_transform(X)), X)

print(model2.fit().summary())

print("Polynomial Regression degree 2 R^2")
print(r2_score(y, poly_regr.predict(poly_features.fit_transform(X))))



#%% 4__Support Vector Regression(SVR)
#Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler2 = StandardScaler()
y_scaled = scaler2.fit_transform(y)

#%% Create Model(support vector regression)

from sklearn.svm import SVR

sv_regr = SVR(kernel = "rbf")
sv_regr.fit(X_scaled, y_scaled)

print("SVR OLS")
model3 = sm.OLS(sv_regr.predict(X_scaled), X_scaled)

print(model3.fit().summary())

print("SVR R^2")
print(r2_score(y_scaled, sv_regr.predict(X_scaled)))

#%% 5__Decision Tree
#Create Model(Decision Tree)
from sklearn.tree import DecisionTreeRegressor

dt_regr = DecisionTreeRegressor(random_state=42)
dt_regr.fit(X, y)

print("Decision Tree OLS")
model4 = sm.OLS(dt_regr.predict(X), X)

print(model4.fit().summary())

print("Decision Tree R^2")
print(r2_score(y, dt_regr.predict(X)))

#%% 6__Random Forest
# Create Model
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regr.fit(X, y.ravel())

print("Random Forest OLS = >")
model5 = sm.OLS(rf_regr.predict(X), X)

print(model5.fit().summary())

print("Random Forest R^2")
print(r2_score(y, rf_regr.predict(X)))

