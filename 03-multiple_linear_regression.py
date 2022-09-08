# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/veriler.csv")

print(dataset)

#%%1_Data preprocessing
# categorical data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
dataset["cinsiyet"] = dataset[["cinsiyet"]].apply(le.fit_transform)

ohe = OneHotEncoder()
country = dataset[["ulke"]]
country = ohe.fit_transform(country).toarray()
country_df = pd.DataFrame(country, columns=["fr", "tr", "us"])

print(country_df)


# collecting dataset
df = pd.concat([dataset.drop("ulke", axis=1), country_df], axis=1)

print(df)

#%% train-test split
y = df["boy"] 
X = df.drop("boy", axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

#%% scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Create model
from sklearn.linear_model import LinearRegression

lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

print("Accuracy:", lin_regr.score(X_test, y_test))

#%% Predictions
y_pred = lin_regr.predict(X_test)

plt.plot(y_test, y_test, "r--")
plt.scatter(y_test, y_pred, color="blue")
plt.show()

#%% Bacward elimination
import statsmodels.api as sm

# X = np.append(arr=np.ones((22, 1)).astype(int), values=X, axis=1)
X = sm.add_constant(X)

#%% Step1
X_opt = X.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
#X_opt = np.array(X_opt, dtype=float)

results = sm.OLS(y, X_opt).fit()

print(results.summary())
# print(model.params)

#%% Step2
X_opt = X.iloc[:, [0, 1, 3, 4, 5, 6]]
results = sm.OLS(y, X_opt).fit()

print(results.summary())