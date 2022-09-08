# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:36:52 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

# Read csv with url
# url = "https://bilkav.com/satislar.csv"
# dataset = pd.read_csv(url)

# print(dataset)
# dataset.to_csv("satislar.csv", index=False)

dataset = pd.read_csv("satislar.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

#%% Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                  test_size=0.33,
                                                  random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X_train,  y_train)

print(lin_regr.score(X_test, y_test))

#%% Predictions
y_pred = lin_regr.predict(X_test)

#%% Save Model 
import pickle

filename = "finalized_model.sav"
pickle.dump(lin_regr, open(filename, "wb"))

#%% Load Model
loaded_model = pickle.load(open(filename, "rb"))

y_pred2 = loaded_model.predict(X_test)