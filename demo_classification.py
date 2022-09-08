# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


dataset = pd.read_excel("datasets/maliciousornot.xlsx")

print(dataset)
print(dataset.isnull().sum())
#sns.countplot(dataset["Type"])

#%% Data Preprocessing
y = dataset["Type"].values
X = dataset.drop("Type", axis=1).values

#%%Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                  test_size=0.33,
                                                  random_state=42)

#5_Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 1_Create Model(Logistic Regression)
from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(random_state=42)
log_regr.fit(X_train, y_train)

#%% Predictions
y_pred = log_regr.predict(X_test)

#%% Confusion Matrix
print("LogisticRegression")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(log_regr.score(X_test, y_test))
# print(classification_report(y_test, y_pred))

#%% 2_Create Model(k-nearest neighbors(k-NN))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("k-NN")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(knn.score(X_test, y_test))
#print(classification_report(y_test, y_pred))

#%% 3_Create Model(Support Vector Classification)
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
#print(pd.Series(y_pred).value_counts())

print("SVC")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(svc.score(X_test, y_test))
# print(classification_report(y_test, y_pred))

#%% 4_Create Model(Naive Bayes-BernoulliNB)
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

print("Bernoulli Naive Bayes")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(bnb.score(X_test, y_test))
# print(classification_report(y_test, y_pred))


#%% 5_Create Model(Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy", random_state=0)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print("DecisionTreeClassifier")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(dtc.score(X_test, y_test))
# print(classification_report(y_test, y_pred))

#%% 6_Create Model(Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=8, criterion="gini", random_state=42)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("RandomForestClassifier")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(rfc.score(X_test, y_test))
# print(classification_report(y_test, y_pred))

#%% 7_Create Model(XGBClassifier)
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

print("XGBoost")
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(xgb_clf.score(X_test, y_test))
# print(classification_report(y_test,  y_pred))

#%% ROC curve
fig = plt.figure(figsize=(7, 4))
my_axes = fig.add_axes([0, 0, 1, 1])

metrics.plot_roc_curve(log_regr, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(knn, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(svc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(bnb, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(dtc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(rfc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(xgb_clf, X_test, y_test, ax=my_axes)
my_axes.grid()




