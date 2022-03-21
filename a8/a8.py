#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:35:14 2021

@author: nickfang
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import os as os

os.chdir('/Users/nickfang/Desktop/IMT 574/a8')
directory = os.getcwd()

df = pd.read_csv('battles.csv')
df = df['Battle'].str.split(',', expand=True)
df.columns = [['Battle', 'Year', 'Portuguese ships', 'Dutch ships', 'English ships', 'The ratio of Portuguese to Dutch/British ships', 'Spanish Involvement', 'Portuguese outcome']] 


df = pd.DataFrame({
    'Battles': [
'Bantam                        1601       6       3       0       2       0       0', 
'Malacca Strait                1606      14      11       0   1.273       0       0',
'Ilha das Naus                 1606       6       9       0   0.667       0      -1',
'Pulo Butum                    1606       7       9       0   0.778       0       1',
'Surrat                        1615       6       0       4     1.5       0       0',
'Ilha das Naus                 1615       3       5       0     0.6       0      -1',
'Jask                          1620       4       0       4       1       0       0',
'Hormuz                        1622       6       0       5     1.2       0      -1',
'Mogincoal Shoals              1622       4       4       2   0.667       0      -1',
'Hormuz                        1625       8       4       4       1       0       0',
'Goa                           1636       6       4       0     1.5       0       0',
'Goa                           1637       6       7       0   0.857       0       0',
'Goa                           1638       6       8       0    0.75       0       1',
'Colombo                       1654       5       3       0   1.667       0       1',
'Goa                           1658       9       9       0       1       0       0',
'Invincible Armada             1588      69       0      31   2.226       1      -1',
'Bahia                         1624       4      13       0   0.308       1      -1',
'Bahia                         1625      35      20       0    1.75       1       1',
'Bahia                         1627       4      10       0     0.4       1      -1',
'Recife                        1630       9      60       0    0.15       1      -1',
'Abrolhos                      1631      17      16       0   1.063       1       0',
'Bahia                         1636       2       8       0    0.25       1       0',
'Dunas                         1639      51      11       0   4.636       1       0',
'Dunas                         1639      38     110       0   0.345       1      -1',
'Paraiba                       1640      16      30       0   0.533       1       0',
'Tamandare                     1645       6       7       0   0.857       1      -1',
'Recife                        1653      14       5       0     2.8       1       1',
'Lisbon                        1657       7      10       0     0.7       1       0'
]})


#df['purchase'].astype(str).astype(int)
df['Battle'].astype(str)

for i in range(0, len(df.columns)):
    df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], errors='ignore')

#dataTypeSeries = empDfObj.dtypes
dataTypeSeries = df.dtypes

#battles.columns = battles.columns.str.replace(' ', '')

#battles['Battle'] = battles['Battle'].str.strip()


X = df[['Portuguese ships', 'Dutch ships', 'English ships', 'Spanish Involvement']]
y = df[['Portuguese outcome']]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)

model = SVC(kernel='linear')
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

tree_predictions = dtree.predict(X_test)
print(accuracy_score(y_test, tree_predictions))
print(confusion_matrix(y_test, tree_predictions))
print(classification_report(y_test,tree_predictions))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print(accuracy_score(y_test, knn_predictions))
print(confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test,knn_predictions))

















