# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("maliciousornot.xlsx")
print(dataset)
print(dataset.isnull().sum())
#sns.countplot(dataset["Type"])

#%% Data Preprocessing

y = dataset["Type"].values
X = dataset.drop("Type", axis=1).values


#%%Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=0)

#5_Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% --My practice --
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier
# 
# rfc = RandomForestClassifier(n_estimators=5, criterion="gini", random_state=0)
# 
# from sklearn.model_selection import cross_val_score, GridSearchCV
# 
# # scores = cross_val_score(rfc, X_train, y_train, cv=5)
# # print(scores.mean())
# 
# parameters = [{"n_estimators": [i for i in range(1, 25)], "criterion": ["gini"]},  
#               {"n_estimators": [i for i in range(1, 25)], "criterion":["entropy"]}]
# grid_search = GridSearchCV(estimator=rfc, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# 
# print(grid_search.best_score_)
# print(grid_search.best_estimator_)
# 
# grid_search_df = pd.DataFrame(grid_search.cv_results_)
# 
# =============================================================================

#%% 1_Create Model(Logistic Regression)

from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(random_state=0)
log_regr.fit(X_train, y_train)

#%% Predictions

y_pred = log_regr.predict(X_test)

#%% Confusion Matrix

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

print("LogisticRegression")
print(log_regr.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm=confusion_matrix(y_test, y_pred)
print(cm)

# metrics.plot_roc_curve(log_regr, X_test, y_test)
# plt.show()

#%% 2_Create Model(k-nearest neighbors(k-NN))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("k-NN")
print(knn.score(X_test, y_test))
#print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)


#%% 3_Create Model(Support Vector Classification)

from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)


y_pred = svc.predict(X_test)
#print(pd.Series(y_pred).value_counts())

print("SVC")
print(svc.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% 4_Create Model(Naive Bayes-BernoulliNB)

from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)

print("Bernoulli Naive Bayes")
print(bnb.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% 5_Create Model(Decision Tree Classifier)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy", random_state=0)
dtc.fit(X_train, y_train)


y_pred = dtc.predict(X_test)

print("DecisionTreeClassifier")
print(dtc.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% 6_Create Model(Random Forest Classifier)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=8, criterion="gini", random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("RandomForestClassifier")
print(rfc.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%% 7_Create Model(XGBClassifier)

import xgboost as xgb

clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("XGBoost")
print(clf.score(X_test, y_test))
#print(classification_report(y_test,  y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
#%%
from sklearn import metrics

fig = plt.figure(figsize=(7, 4))
my_axes = fig.add_axes([0, 0, 1, 1])

metrics.plot_roc_curve(log_regr, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(knn, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(svc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(bnb, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(dtc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(rfc, X_test, y_test, ax=my_axes)
metrics.plot_roc_curve(clf, X_test, y_test, ax=my_axes)
my_axes.grid()




