# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("Social_Network_Ads.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

#%% Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model
from sklearn.svm import SVC

svc = SVC(kernel="rbf", random_state=1)
svc.fit(X_train, y_train)

#%% Predictions
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

#%% 1__k-fold cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=4)

print("mean", scores.mean())
print("std", scores.std())
print(scores)

#%% 2__GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = [
    {"C":[1, 2, 3, 4, 5], "kernel":["linear"]}, 
    {"C":[1, 10, 100, 1000], "kernel":["rbf"], "gamma":[1, 0.5, 0.1, 0.01, 0.001]}]

grid_search = GridSearchCV(estimator=svc, 
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=5,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_
best_params = grid_search.best_params_

print("best_score", best_score)
print("best_params", best_params)

#print(grid_search.cv_results_)
df = pd.DataFrame(grid_search.cv_results_)

print(df[["params", "mean_test_score", "rank_test_score"]].sort_values("rank_test_score").head())
