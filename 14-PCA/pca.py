# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt


dataset = pd.read_csv("Wine.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
X = dataset.iloc[:, :13].values
y = dataset.iloc[:, 13].values

#%% Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42)

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Principal component analysis (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#%% Create Model
from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(random_state=42)
log_regr.fit(X_train, y_train)

log_regr_pca = LogisticRegression(random_state=42)
log_regr_pca.fit(X_train_pca, y_train)

#%% Predictions
y_pred = log_regr.predict(X_test)
y_pred2 = log_regr_pca.predict(X_test_pca)

from sklearn.metrics import confusion_matrix

print("Original")
cm = confusion_matrix(y_test, y_pred)

print(cm)

print("with PCA")
cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)

print("original vs PCA")
cm3 = confusion_matrix(y_pred, y_pred2)

print(cm3)

#%% Linear Discriminant Analysis(LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

#%% Create Model
from sklearn.linear_model import LogisticRegression

log_regr_lda = LogisticRegression(random_state=42)
log_regr_lda.fit(X_train_lda, y_train)

#%% Predictions
y_pred3 = log_regr_lda.predict(X_test_lda)

from sklearn.metrics import confusion_matrix

print("with LDA")
cm4 = confusion_matrix(y_test, y_pred3)

print(cm4)

print("original vs LDA")
cm3 = confusion_matrix(y_pred,  y_pred3)

print(cm3)
