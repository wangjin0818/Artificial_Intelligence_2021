import os
import sys
import logging

import numpy as np
import pandas as pd

import tensorflow as tf

epoch = 1000
rate = 0.001

# load data
file_name = os.path.join('data', 'one_variable.txt')
data = pd.read_table(file_name, sep='\t', header=None, quoting=3)
# print(data)

print(data[0], data[1])

train_x = np.array(data[0])
train_y = np.array(data[1])

# tensorflow model
x = tf.placeholder("float")  
y = tf.placeholder("float")  

w = tf.Variable(tf.random_normal([1])) # 生成随机权重，也就是我们的theta_1
b = tf.Variable(tf.random_normal([1])) # theta_2

y_pred = tf.add(tf.multiply(x, w), b)
loss = tf.reduce_sum(tf.pow(y_pred - y, 2))
optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()  
sess.run(init)  
print('w start is ', sess.run(w))  
print('b start is ', sess.run(b))  
for index in range(epoch):  
    #for tx,ty in zip(train_x,train_y):  
        #sess.run(optimizer,{x:tx,y:ty})  
    sess.run(optimizer, {x: train_x, y: train_y}) 

    if index % 10 == 0:
        print('w is', sess.run(w), ' b is', sess.run(b), ' loss is', sess.run(loss, {x: train_x, y: train_y}))

print('loss is ', sess.run(loss, {x: train_x, y: train_y})) 
print('w end is ',sess.run(w))  
print('b end is ',sess.run(b)) 

print('y_pred is ', sess.run(y_pred, {x: [0.25]}))