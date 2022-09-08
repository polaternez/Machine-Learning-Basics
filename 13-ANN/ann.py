# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


dataset = pd.read_csv("Churn_Modelling.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
# dataset2 = dataset.iloc[:, 3:13]
# print(dataset2["Geography"].value_counts())
#######################
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%% Categorical Data Encoding
# from sklearn import preprocessing

# dataset2 = dataset2.apply(preprocessing.LabelEncoder().fit_transform)
# print(dataset2)

# geography = dataset2.iloc[:, 1:2].values

# ohe = preprocessing.OneHotEncoder()
# geography = ohe.fit_transform(geography).toarray()
# print(geography)

# geography = pd.DataFrame(geography, columns=["France", "Germany", "Spain"])
# print(geography)

# result = pd.concat([dataset2.iloc[:,:1],geography,dataset2.iloc[:,2:3],dataset.iloc[:,6:]],axis=1)
# print(result)
################
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X2 = ohe.fit_transform(X)

#%% Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation="relu"))
model.add(Dense(9, activation="relu"))
model.add(Dense(9, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=64, epochs=150,
          validation_data=(X_test, y_test))

#%% Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))