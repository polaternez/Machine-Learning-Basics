# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score


dataset = pd.read_csv("Churn_Modelling.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
# dataset2 = dataset.iloc[:, 3:13]
# print(dataset2["Geography"].value_counts())
#######################
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%%2_ Encoding Categorical Data
# from sklearn import preprocessing

# dataset2 = dataset2.apply(preprocessing.LabelEncoder().fit_transform)
# print(dataset2)

# geography = dataset2.iloc[:, 1:2].values

# ohe = preprocessing.OneHotEncoder()
# geography = ohe.fit_transform(geography).toarray()
# print(geography)

# geography = pd.DataFrame(geography, columns=["France", "Germany", "Spain"])
# print(geography)

# result = pd.concat([dataset2.iloc[:, :1], geography, dataset2.iloc[:, 2:3], dataset.iloc[:, 6:]], axis=1)
# print(result)
################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

le2 = LabelEncoder()
X[:, 2] = le2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer

column_trans = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])], remainder="passthrough")
X2 = column_trans.fit_transform(X)

#%% Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, y, 
                                                    test_size=0.20,
                                                    random_state=42)

#%% Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model

from xgboost import XGBClassifier

params = {
    'gamma': 6,
    'learning_rate': 0.1,
    'max_depth': 12,
    'min_child_weight': 1,
    'subsample': 1
}

model = XGBClassifier(**params)
scores = cross_val_score(model, X_train, y_train.ravel(), cv=5)

print(scores)
print(scores.mean())
print(scores.std())

#%% Predictions
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

print(cm)
print(classification_report(y_test, y_pred))

#%% Grid Search
# =============================================================================
# 
# parameters = [{
#     "learning_rate": [0.3, 0.1, 0.03, 0.01],
#     "max_depth": [6, 12, 18],
#     "gamma": [0, 6, 12],
#     "min_child_weight":[1, 4, 8],
#     "subsample":[1]
#     }]
# 
# grid_search = GridSearchCV(model, param_grid=parameters, scoring="accuracy",
#                            cv=5, n_jobs=-1)
# 
# grid_search.fit(X_train, y_train)
# 
# print("best score", grid_search.best_score_)
# print(grid_search.best_params_)
# =============================================================================




