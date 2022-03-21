#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:48:01 2021

@author: nickfang
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import os as os
from sklearn.tree import export_graphviz



directory = os.getcwd()
os.chdir('/Users/nickfang/Desktop/IMT 574/a4')
data = pd.read_csv('blues_hand.csv')

X = data[['handPost', 'thumbSty', 'region']]
y = data['brthYr']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

