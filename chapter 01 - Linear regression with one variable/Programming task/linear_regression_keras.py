import os
import sys
import logging

import numpy as np
import pandas as pd

from keras.layers import Dense, Input
from keras.models import Model

epoch = 50
batch_size = 2

# load data
file_name = os.path.join('data', 'one_variable.txt')
data = pd.read_table(file_name, sep='\t', header=None, quoting=3)
# print(data)
# print(data[0], data[1])


train_x = np.array(data[0])
train_y = np.array(data[1])

x = Input(shape=(1, ))
y = Dense(1, activation='linear')(x)

model = Model(x, y)
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,)

print(model.layers[1].get_weights())
print(model.predict(np.array([[0.25]])))