#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 12:53:43 2021

@author: nickfang
"""
import os as os
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

directory = os.getcwd()
os.chdir('/Users/nickfang/Desktop/IMT 574/a2')

data = pd.read_csv('airline_costs.csv')
# y = c + mX1 + mX2

data.columns = ['Airline',
                'Length of flight (miles)',
                'Speed of plane (miles per hour)', 
                'Daily flight time per plane (hours)',
                'Population served (1000s)',
                'Total operating cost (cents per revenue ton-mile)',
                'Revenue tons per aircraft mile',
                'Ton-mile load factor (proportion)',
                'Available capacity (tons per mile)',
                'Total assets ($100,000s)',
                'Investments and special funds ($100,000s)',
                'Adjusted assets ($100,000s)']
print(data)

#Problem 1
X = data[['Length of flight (miles)','Daily flight time per plane (hours)']]
y = data ['Population served (1000s)']

regr = linear_model.LinearRegression()
regr.fit(X, y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X) 
lr_model = sm.OLS(y, X).fit()
predictions = lr_model.predict(X)
print_model = lr_model.summary()
print(print_model)

# y = -8926.321 + 193.699 * X1 + -346.445 * X2
# predicted value = 27,319
predicted_value = -8926.321 + 193.699 * 200 + -346.445 * 7.2
print(predicted_value)

#Problem 2
#y = c + mX1
X = data['Population served (1000s)']
y = data['Total assets ($100,000s)']
X = sm.add_constant(X)

model2 = sm.OLS(y, X).fit()
print_model2 = model2.summary()
print(print_model2)

# y = -91.085 + 0.022X
# predicted assets is 446,508.915
predicted_assets = -91.085 + 0.022 * 20300000
print('Predictions for total assets given 20,300,000 customers is', 
      predicted_assets)

#Problem 3
kangarooData = pd.read_csv('kangaroos.csv')

X = df[['X']]
X = sm.add_constant(X).to_numpy()
y = df[['Y']].to_numpy()

def grad_descent (X, y, alpha, epsilon):
    iteration = [0]
    i = 0
    theta = np.ones(shape=(len(df.columns),0))
    cost = [np.transpose(X @ theta - y) @ (X @ theta - y)]
    delta = 1
    while (delta>epsilon):
        theta = theta - alpha*((np.transpose(X)) @ (X @ theta - y))
        cost_val = (np.transpose(X @ theta - y)) @ (X @ theta - y)
        cost.append(cost_val)
        delta = abs(cost[i+1]-cost[i])
        if ((cost[i+1]-cost[i]) > 0):
            print("error")
            break
        iteration.append[i]
        i += 1
    print("Completed in %d iterations" %(i))
    return theta

theta = grad_descent(X = preprocessing.scale(X), y=y, alpha=0.01,
                     epsilon = 10**-10)
print(theta)

#function error, will continue reaidng about gradient descent and get more 
#comfortable with it on python
        
import seaborn as sns


lr_model = sm.OLS(kangarooData.Y, kangarooData.X).fit()
lr_model.summary()
sns.lmplot(x = 'X', y = 'Y', data = kangarooData)

def cost(X, y, theta):
    return(sum(np.matmul(X, theta) - y)**2 / (2*len(y)))


theta = np.matrix([[0],[0]])
num_iterations = 10
alpha = 0.01

# Store the history
cost_history = [float(num_iterations)]
theta_history = [[num_iterations]]

X = np.append(np.ones([len(kangarooData),1]), np.array(kangarooData.X).reshape(len(kangarooData.X),1), 1)

for i in range(1, num_iterations+1):
    error = np.matmul(X, theta) - np.array(kangarooData.Y).reshape(len(kangarooData.Y),1)
    delta = np.matmul(np.transpose(X), error) / len(kangarooData.Y)
    theta = theta - alpha * delta
    #print(theta)
    cost_history.append(cost(X, np.array(kangarooData.Y).reshape(len(kangarooData.Y),1), theta))
    theta_history.append([theta])
    

import matplotlib.pyplot as plt

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

plt.figure()
plt.scatter(kangarooData.X, kangarooData.Y)
abline(float(theta_history[1][0][0]), float(theta_history[1][0][1]))
abline(float(theta_history[1][0][0]), float(theta_history[2][0][1]))
abline(float(theta_history[1][0][0]), float(theta_history[3][0][1]))
abline(float(theta_history[1][0][0]), float(theta_history[4][0][1]))
plt.show()

plt.plot(cost_history)







