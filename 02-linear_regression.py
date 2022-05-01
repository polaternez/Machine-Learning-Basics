# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt


dataFrame = pd.read_csv("satislar.csv")
print(dataFrame)

aylar = dataFrame[["Aylar"]]
print(aylar)

satislar = dataFrame[["Satislar"]]
print(satislar)


#Train/Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,
                                                    test_size=0.33,
                                                    random_state=42)

#Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Create Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)



#Predictions

predictions = lr.predict(X_test)

#plot
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(X_test))
plt.title("Aylara Göre Satislar")
plt.xlabel("Aylar")
plt.ylabel("Satislar")