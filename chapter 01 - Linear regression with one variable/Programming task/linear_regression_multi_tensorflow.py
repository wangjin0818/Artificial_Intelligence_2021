import os
import sys
import logging

import numpy as np
import pandas as pd

import tensorflow as tf

epoch = 1000
rate = 0.5

# load data
file_name = os.path.join('data', 'multiple_variable.txt')
data = pd.read_table(file_name, sep='\t', header=None, quoting=3)

train_x = []; train_y = []
for i in range(len(data[0])):
    train_x.append([data[0][i], data[1][i], data[2][i]])
    train_y.append([data[3][i]])

train_x = np.array(train_x)
train_y = np.array(train_y)

# tensorflow model
x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None, 1])  

# w = tf.Variable(tf.random_normal([1])) # 生成随机权重，也就是我们的theta_1
w = tf.Variable(tf.random_uniform([3, 1], -1., 1.))
b = tf.Variable(tf.zeros([1])) # theta_0

y_pred = tf.matmul(x, w) + b
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()  
sess.run(init)  
print('w start is ', sess.run(w))  
print('b start is ', sess.run(b))  
for index in range(epoch):  
    sess.run(optimizer, {x: train_x, y: train_y}) 

    if index % 10 == 0:
        print('w is', sess.run(w), ' b is', sess.run(b), ' loss is', sess.run(loss, {x: train_x, y: train_y}))

print('loss is ', sess.run(loss, {x: train_x, y: train_y})) 
print('w end is ',sess.run(w))  
print('b end is ',sess.run(b)) 
print('y_pred is ', sess.run(y_pred, {x: [[0.25, 0.25, 0.25]]}))