# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.metrics import r2_score


dataset = pd.read_csv("datasets/maaslar.csv")

print(dataset)

#%%Train/Test Split
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

#%% 1__Create Model(linear regression)
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X, y)

# Predictions(linear regression)
plt.scatter(X, y, color="red")
plt.plot(X, lin_regr.predict(X), "b")
plt.show()

print("Linear Regression  R^2")
print(r2_score(y, lin_regr.predict(X)))

#%% 2a__Create Model(polynomial regression degree 2)
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

print(X_poly)

#Modelling
poly_regr = LinearRegression()
poly_regr.fit(X_poly, y)

# Predictions(polynomial regression)
plt.scatter(X, y, color="red")
plt.plot(X, poly_regr.predict(poly_features.fit_transform(X)), "b")
plt.show()

print("Polynomial Regression degree 2 R^2")
print(r2_score(y, poly_regr.predict(poly_features.fit_transform(X))))

#%% 2b__Create Model(polynomial regression degree 4)
from sklearn.preprocessing import PolynomialFeatures

poly_features2 = PolynomialFeatures(degree=4)
X_poly = poly_features2.fit_transform(X)

print(X_poly)

poly_regr2 = LinearRegression()
poly_regr2.fit(X_poly, y)

# Predictions(polynomial regression2)
plt.scatter(X, y, color="red")
plt.plot(X, poly_regr2.predict(poly_features2.fit_transform(X)), "b")
plt.show()

print("Polynomial Regression degree 4 R^2")
print(r2_score(y, poly_regr2.predict(poly_features2.fit_transform(X))))

#%% 3__Predictions

print(lin_regr.predict([[11]]))
print(lin_regr.predict([[6.6]]))

print(poly_regr.predict(poly_features.fit_transform([[11]])))
print(poly_regr.predict(poly_features.fit_transform([[6.6]])))

print(poly_regr2.predict(poly_features2.fit_transform([[11]])))
print(poly_regr2.predict(poly_features2.fit_transform([[6.6]])))

#%% 4__Support Vector Regression(SVR)
#Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler2 = StandardScaler()
y_scaled = scaler2.fit_transform(y)

#%% Create Model(support vector regression)
from sklearn.svm import SVR

sv_regr = SVR(kernel="rbf")
sv_regr.fit(X_scaled, y_scaled.ravel())

plt.scatter(X_scaled, y_scaled, color="blue")
plt.plot(X_scaled, sv_regr.predict(X_scaled), "r")
plt.show()

pred = sv_regr.predict(scaler.transform([[11]]))
pred2 = sv_regr.predict(scaler.transform([[6.6]]))

print(scaler2.inverse_transform(pred.reshape(1, -1)))
print(scaler2.inverse_transform(pred2.reshape(1, -1)))

print("SVR R^2")
print(r2_score(y_scaled, sv_regr.predict(X_scaled)))

#%% 5__Decision Tree
#Create Model(Decision Tree)
from sklearn.tree import DecisionTreeRegressor

dt_regr = DecisionTreeRegressor(random_state=42)
dt_regr.fit(X, y)

Z = X + 0.5
K = X - 0.4

plt.scatter(X, y, color="blue")
plt.plot(X, dt_regr.predict(X), "r")

plt.plot(X, dt_regr.predict(Z), "g")
plt.plot(X, dt_regr.predict(K), "black")
plt.show()

print(dt_regr.predict([[11]]))
print(dt_regr.predict([[6.6]]))

print("Decision Tree R^2")
print(r2_score(y, dt_regr.predict(X)))

#%% 6__Random Forest
# Create Model
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regr.fit(X, y.ravel())

plt.scatter(X, y, color="blue")
plt.plot(X, rf_regr.predict(X), "r", label="X")

plt.plot(X, rf_regr.predict(Z), "g", label="Z")
plt.plot(X, rf_regr.predict(K), "yellow", label="K")
plt.legend()
plt.show()

print(rf_regr.predict([[11]]))
print(rf_regr.predict([[6.6]]))

#%% 7__Evaluation of Prediction
print("Random Forest R^2")
print(r2_score(y, rf_regr.predict(X)))

print(r2_score(y, rf_regr.predict(Z)))
print(r2_score(y, rf_regr.predict(K)))