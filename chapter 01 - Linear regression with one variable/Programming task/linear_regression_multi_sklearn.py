import os
import sys
import logging

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.linear_model import LinearRegression

# load data
file_name = os.path.join('data', 'multiple_variable.txt')
data = pd.read_table(file_name, sep='\t', header=None, quoting=3)
# print(data)
# print(data[0], data[1])

train_x = []
for i in range(len(data[0])):
    train_x.append([1., data[0][i], data[1][i], data[2][i]])

# train_x = data[0]

train_x = np.array(train_x)
train_y = np.array(data[3])

lr = LinearRegression()
lr.fit(train_x, train_y)

print(lr.coef_)
print(lr.intercept_)
print(lr.predict([[1., 0.25, 0.25, 0.25]]))