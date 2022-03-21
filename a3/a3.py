#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:56:07 2021

@author: nickfang
"""

import pandas as pd 
import os as os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



directory = os.getcwd()
os.chdir('/Users/nickfang/Desktop/IMT 574/a3')

data = pd.read_csv('quality.csv')
print(data.shape)
print(list(data.columns)) 

#Create a logistics regression model to predict the class label from the 
#first eight attributes of the question set.

data['label'] = np.where(data['label'] == 'B', 1, 0)

X = data[['num_words', 'num_characters', 'num_misspelled', 
          'bin_end_qmark', 'num_interrogative', 'bin_start_small', 
          'num_sentences', 'num_punctuations']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test) 

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#2.	Try doing the same using two different subsets (your choice) of those eight attributes.

data = pd.read_csv('wine.csv')
X = data[['volatile_acidity', 'residual_sugar', 'chlorides', 'density', 'alcohol']]
y = data['high_quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

neighbors = [2,3,4,5,6,7,8,9,10]
accuracies = [83.08, 82, 82.05, 81.38, 81.74, 81.28, 80.87, 81.28, 81.28]
plt.plot(neighbors, accuracies, color = 'red', marker = 'o')
plt.title('Accuracy of KNeighbors 2:10')
plt.xlabel('K')
plt.ylabel('Accuracy Score')
plt.show()

