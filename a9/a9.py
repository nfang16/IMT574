#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 21:43:21 2021

@author: nickfang
"""

import pandas as pd
import os as os
from sklearn.model_selection import train_test_split

os.chdir('/Users/nickfang/Desktop/IMT 574/a9')
directory = os.getcwd()


dataset = pd.read_csv('Faults.csv')
headers = list(dataset.columns.values)

headers_input = headers[:-7]
headers_output = headers[-7:]

print('features for input X:', headers_input)
print('classes for output Y:', headers_output)

input_x = dataset[headers_input]
output_y = dataset[headers_output]

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
np_scaled = min_max_scaler.fit_transform(input_x)
headers_27 = list(input_x.columns.values)
input_x_27 = pd.DataFrame(np_scaled)
input_x_27.columns = headers_27
input_x_27[0:3]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
targets=(output_y.iloc[:,:]==1).idxmax(1)
print(targets.value_counts())
Y=le.fit_transform(targets)
print(len(Y))

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y_one_hot = encoder.fit_transform(Y)
print(Y_one_hot)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
ros = RandomOverSampler(random_state=0)
X = input_x_norm #df_normalized
ros.fit(X, Y)
X_resampled, y_resampled = ros.fit_sample(X, Y)
print('Amount of elements before:', len(X))
print('Amount of elements after:', len(X_resampled))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 Y_one_hot,
                                                 test_size = 0.3,#%70 train, 30% test
                                                 random_state = 3)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=22))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=2000,
          batch_size=128)


























