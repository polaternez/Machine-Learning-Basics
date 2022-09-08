# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
plt.style.use("ggplot")

dataset = pd.read_excel("datasets/merc.xlsx")

print(dataset)

sns.distplot(dataset["price"])
# print(dataset.isnull().sum())

#%% Data processing
dataset = dataset.sort_values("price", ascending=False).iloc[100:, :]
sns.distplot(dataset["price"])

#%% Train/Test Split
X = dataset.drop(["price", "transmission"], axis=1).values
y = dataset["price"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=73)

#%% Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 1_Create model(linear regression)
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

# predictions(linear regression)
y_pred = lin_regr.predict(X_test)

# plot
plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

print("Linear Regression  R^2", r2_score(y_test, y_pred.ravel()))
print(lin_regr.score(X_test, y_test))

#%%
import statsmodels.api as sm

# X_1 = np.append(arr=np.ones((13019, 1)).astype(int), values=X, axis=1)
X_1 = sm.add_constant(X)
X_opt = X_1[:,[0, 1, 2, 3, 4, 5]]

results = sm.OLS(y, X_opt).fit()

print(results.summary())

#%% 2a_Create model(polynomial regression)
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)

print(X_poly)

poly_regr = LinearRegression()
poly_regr.fit(X_poly, y_train)

# predictions(polynomial regression)
y_pred = poly_regr.predict(poly_features.fit_transform(X_test))

# plot
plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

print("Polynomial Regression degree 2 R^2")
print(r2_score(y_test, y_pred))


#%% 2b_Create model(polynomial regression2)
from sklearn.preprocessing import PolynomialFeatures

poly_features2 = PolynomialFeatures(degree=4)
X_poly = poly_features2.fit_transform(X_train)

print(X_poly)

poly_regr2 = LinearRegression()
poly_regr2.fit(X_poly, y_train)

# predictions(polynomial regression)
y_pred = poly_regr2.predict(poly_features2.fit_transform(X_test))

# plot
plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

print("Polynomial Regression degree 4 R^2")
print(r2_score(y_test, y_pred))

#%% 3_Create model(support vector regression)
from sklearn.preprocessing import StandardScaler

scaler2 = StandardScaler()
y_train_scaled = scaler2.fit_transform(y_train.reshape(len(y_train), 1))
y_test_scaled = scaler2.transform(y_test.reshape(len(y_test), 1))

from sklearn.svm import SVR

sv_regr = SVR(kernel="rbf")
sv_regr.fit(X_train, y_train_scaled.ravel())

# predictions(polynomial regression)
y_pred = sv_regr.predict(X_test)

# plot
plt.plot(y_test_scaled, y_test_scaled, "r--")
plt.scatter(y_test_scaled, y_pred, color="blue")
plt.show()

print("SVR R^2")
print(r2_score(y_test_scaled, y_pred))

#%% 5_Create model(Decision Tree)
from sklearn.tree import DecisionTreeRegressor

dt_regr = DecisionTreeRegressor(random_state=42)
dt_regr.fit(X_train, y_train)

# predictions
y_pred = dt_regr.predict(X_test)

# plot
plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

print("Decision Tree R^2")
print(r2_score(y_test, y_pred))

#%% 6_Create model(Random Forest)
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor(n_estimators=28, random_state=42)
rf_regr.fit(X_train, y_train)

# predictions
y_pred = rf_regr.predict(X_test)

# plot
plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

print("Random Forest R^2", r2_score(y_test, y_pred))
print("Score", rf_regr.score(X_test, y_test))

#%%
# for i in range(1, 30):
#     rf_regr = RandomForestRegressor(n_estimators=i, random_state=42)
#     rf_regr.fit(X_train, y_train)
#     y_pred = rf_regr.predict(X_test)
#     print(f"n_estimator:{i}->>>R^2 : {r2_score(y_test, y_pred)}")

#%% Prediction with new data
dataset2 = dataset.drop(["price", "transmission"], axis=1)
new_car = dataset2.iloc[:5, :].values
new_car_scaled = scaler.transform(new_car)

new_car_pred = rf_regr.predict(new_car_scaled)

print(new_car_pred)
