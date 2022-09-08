# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


comments_df = pd.read_excel("Restaurant_Reviews.xlsx")

print(comments_df)

#%% Data Preprocessing
#RegEx module 
import re 

# Natural Language Toolkit
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

ps = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")

comments_cleaned = []

for i in range(len(comments_df)):
    comment = re.sub("[^a-zA-Z]", " ", comments_df["review"][i]) 
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if word not in set(stopwords.words("english"))]
    comment = " ".join(comment)
    comments_cleaned.append(comment)
    
#%% Feature Extraction
# Bag of Words(BOW)
from sklearn.feature_extraction.text import  CountVectorizer

cv = CountVectorizer(max_features=2000)
comments_count = cv.fit_transform(comments_cleaned).toarray()

# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
comments_tfidf = tfidf_trans.fit_transform(comments_count).toarray()
y = comments_df.iloc[:,1].values

#%% Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(comments_tfidf, y,
                                                    test_size=0.20,
                                                    random_state=123)

#%% Create model
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("GaussianNB score:", gnb.score(X_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
# print(classification_report(y_test, y_pred))

#%%
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=17, criterion="gini", random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("random forest score:", rfc.score(X_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#%%
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="gini", random_state=0)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print("decision tree score:", dtc.score(X_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#%%
import xgboost as xgb

clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("xgboost score:", clf.score(X_test, y_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#%% Create model
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

print("MultinomialNB score:", mnb.score(X_test, y_test))

# confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)