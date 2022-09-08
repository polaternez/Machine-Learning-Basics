# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
from sklearn.metrics import r2_score

dataset = pd.read_csv("datasets/tennis.csv")

print(dataset)

#%%1_Missing Values
dataset.info()

#%%2_Categorical Data Encoding
from sklearn import preprocessing

dataset2 = dataset.apply(preprocessing.LabelEncoder().fit_transform)

print(dataset2)

outlook = dataset2.iloc[:, :1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

print(outlook)

outlook = pd.DataFrame(outlook, columns=["overcast", "rainy", "sunny"])

print(outlook)

result = pd.concat([outlook, dataset.iloc[:, 1:3], dataset2.iloc[:, -2:]], axis=1)

print(result)


#%%4_Train/Test Split
X = result.drop("play", axis=1).values
y = result["play"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.33,
                                                    random_state = 42)

#5_Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(X_train, y_train)

print(regr.score(X_test, y_test))

#%% Predictions

y_pred = regr.predict(X_test)

#%% Bacward Elimination

import statsmodels.api as sm

#X = np.append(arr=np.ones((14, 1)).astype(int), values=X, axis=1)
X = sm.add_constant(X)
X_1 = X[:, [0, 1, 2, 3, 4, 5]]
X_1 = np.array(X_1, dtype=float)
model = sm.OLS(y, X_1).fit()

print(model.summary())

#%%

X_1 = X[:, [0, 1, 2, 4, 5]]
X_1 = np.array(X_1, dtype=float)
model = sm.OLS(y, X_1).fit()

print(model.summary())

