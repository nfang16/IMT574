#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 22:16:02 2021

@author: nickfang
"""
import pandas as pd
from sklearn.mixture import GaussianMixture
import os as os


os.chdir('/Users/nickfang/Desktop/IMT 574/a7')
directory = os.getcwd()

life = pd.read_csv('lifecyclesaving.csv')
data = life.iloc[:,1:5]

model = GaussianMixture(n_components=15, init_params='random', max_iter=100)
model.fit(data)

yhat = model.predict(data)
print(yhat)

print(model.aic(data))
print(model.bic(data))
