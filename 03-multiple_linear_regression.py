# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

dataset = pd.read_csv("veriler.csv")

print(dataset)




#%%1_Missing Values

Yas = dataset.iloc[:, 1:4]
print(Yas)
#scaleri-kit learn



#%%2_Categorical Data Encoding

ulke = dataset.iloc[:, 0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:, 0] = le.fit_transform(dataset.iloc[:, 0])
#ulke[:, 0] = le.fit_transform(ulke)
print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

gender = dataset.iloc[:, -1:].values
print(gender)

le = preprocessing.LabelEncoder()

gender[:, 0] = le.fit_transform(dataset.iloc[:, -1:])
#ulke[:, 0] = le.fit_transform(ulke)
print(gender)

ohe = preprocessing.OneHotEncoder()

gender = ohe.fit_transform(gender).toarray()
print(gender)

#%%3_Collecting dataset

result = pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
#result = pd.DataFrame(ulke, columns=["fr", "tr", "us"])
print(result)

result2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy", "kilo", "yas"])
print(result2)



result3 = pd.DataFrame(data=gender[:, :1], index=range(22), columns=["cinsiyet"])
print(result3)

s = pd.concat([result, result2], axis=1)
print(s)

s2 = pd.concat([s, result3], axis=1)
print(s2)

#%%4_Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(s, result3,
                                                    test_size=0.33,
                                                    random_state=42)

#5_scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create Model
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

#%% Predictions

y_pred = lin_regr.predict(X_test)

#%% Other Model

y = s2["boy"].values
X = s2.drop("boy", axis=1).values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                  test_size=0.33,
                                                  random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

y_pred = lin_regr.predict(X_test)
#%% Bacward Elimination

import statsmodels.api as sm

X = np.append(arr=np.ones((22, 1)).astype(int), values=X, axis=1)
#X = sm.add_constant(X)

#%% Step1

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
#X_opt = np.array(X_opt, dtype=float)

results = sm.OLS(y, X_opt).fit()
print(results.summary())
#print(model.params)

#%% Step2
X_opt = X[:, [0, 1, 2, 3, 4, 6]]

results = sm.OLS(y, X_opt).fit()
print(results.summary())