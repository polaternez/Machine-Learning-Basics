# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:10:39 2021

@author: Master
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report

dataset = pd.read_excel("datasets/Iris.xls")

print(dataset)

#%% Iris Plots 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min()-.5, X[:, 0].max()+.5
y_min, y_max = X[:, 1].min()-.5, X[:, 1].max()+.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

#%% Data Preprocessing
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4:].values


#%%Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                    test_size=0.33,
                                                    random_state=42)

#5_Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%% 1__Create Model(Logistic Regression)
from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(random_state=42)
log_regr.fit(X_train,y_train)

# Predictions
y_pred = log_regr.predict(X_test)

#%% Confusion Matrix
print("LogisticRegression")
cm = confusion_matrix(y_test,y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% 2__Create Model(k-nearest neighbors(k-NN))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train,y_train)

# Predictions
y_pred = knn.predict(X_test)

print("k-NN")
cm = confusion_matrix(y_test, y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% 3__Create Model(Support Vector Classification)
from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)

# Predictions
y_pred = svc.predict(X_test)

print("SVC")
cm = confusion_matrix(y_test,y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% 4__Create Model(Naive Bayes-GaussianNB)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)

print("Naive Bayes")
cm = confusion_matrix(y_test, y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% 5__Create Model(Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy", random_state=42)
dtc.fit(X_train,y_train)

# Predictions
y_pred = dtc.predict(X_test)

print("DecisionTreeClassifier")
cm = confusion_matrix(y_test,y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% 6__Create Model(Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=8, criterion="entropy", random_state=42)
rfc.fit(X_train,y_train)

# Predictions
y_pred = rfc.predict(X_test)

print("RandomForestClassifier")
cm = confusion_matrix(y_test, y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%% Receiver operating characteristic (ROC)
from sklearn import metrics

y_proba = rfc.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:,0], pos_label="Iris-setosa")

print(tpr)
print(fpr)
print(thresholds) 

metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# metrics.plot_roc_curve( rfc,X_test,y_test)
# plt.show()
