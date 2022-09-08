# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt 

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

dataset = pd.read_csv("datasets/veriler.csv")

print(dataset)
print(dataset.isnull().sum())

#%% Data Preprocessing
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

#%%Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.33,
                                                    random_state=1)

#5_Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 1__Create Model(Logistic Regression)
from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(random_state=42)
log_regr.fit(X_train, y_train)

# Predictions
y_pred = log_regr.predict(X_test)

#%% Confusion Matrix
print("LogisticRegression")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(log_regr.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))


#%% 2__Create Model(k-nearest neighbors(k-NN))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("k-NN")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(knn.score(X_test,  y_test))
#print(classification_report(y_test,  y_pred))

#%% 3__Create Model(Support Vector Classification)
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("SVC")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(svc.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))

#%% 4__Create Model(Naive Bayes-GaussianNB)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Naive Bayes")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(gnb.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))

#%% 5__Create Model(Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy", random_state=42)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print("DecisionTreeClassifier")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(dtc.score(X_test, y_test))
#print(classification_report(y_test, y_pred))

#%% 6__Create Model(Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=5, criterion="entropy",random_state=42)
rfc.fit(X_train, y_train)

# Evaluate Classification
y_pred = rfc.predict(X_test)

print("RandomForestClassifier")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(rfc.score(X_test, y_test))
#print(classification_report(y_test, y_pred))

#%% Receiver operating characteristic (ROC)
y_proba = rfc.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:, 1], pos_label="k")

print(tpr)
print(fpr)
print(thresholds)
#metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
metrics.plot_roc_curve( rfc, X_test, y_test)
plt.show()

print("roc_auc", metrics.roc_auc_score(y_test, y_proba[:, 1]))

