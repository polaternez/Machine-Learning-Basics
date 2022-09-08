# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt


dataFrame = pd.read_csv("datasets/satislar.csv")

print(dataFrame)

aylar = dataFrame[["Aylar"]]

print(aylar)

satislar = dataFrame[["Satislar"]]

print(satislar)

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(aylar, satislar,
                                                    test_size=0.33,
                                                    random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Create Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predictions

predictions = lr.predict(X_test_scaled)

# Plot
X_train = X_train.sort_index()
y_train = y_train.sort_index()

plt.plot(X_train.values, y_train.values)
plt.plot(X_test.sort_index().values, predictions)
plt.title("Aylara Göre Satislar")
plt.xlabel("Aylar")
plt.ylabel("Satislar")