# -*- coding: utf-8 -*-
"""

@author: jpshlima
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# read data
base = pd.read_csv('petr4_train.csv')
# clean NaNs
base = base.dropna()
# select train columns
base_train = base.iloc[:, 1:2].values
base_maxvalue = base.iloc[:, 2:3].values

# normalizing
norm = MinMaxScaler(feature_range=(0,1))
base_train_norm = norm.fit_transform(base_train)
base_maxvalue_norm = norm.fit_transform(base_maxvalue)

# initialing variables
prev = []
real_price = []
real_maxprice = []

# filling for 90-day prediction
for i in range(90, len(base_train_norm)):
    prev.append(base_train_norm[i-90:i, 0])
    real_price.append(base_train_norm[i, 0])
    real_maxprice.append(base_maxvalue_norm[i, 0])

# adapting to keras formats
prev, real_price, real_maxprice = np.array(prev), np.array(real_price), np.array(real_maxprice)
prev = np.reshape(prev, (prev.shape[0], prev.shape[1], 1))
# combining arrays
comb_price = np.column_stack((real_price, real_maxprice))

# starting regressor
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (prev.shape[1], 1)))
# using dropout to avoid overfitting
regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# final layer using sigmoid this time
regressor.add(Dense(units = 2, activation = 'linear'))

# compiling using adam
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
regressor.fit(prev, comb_price, epochs = 100, batch_size = 32)


# testing phase
# read test data
base_test = pd.read_csv('petr4_test.csv')
real_price_open = base_test.iloc[:,1:2].values
real_price_high = base_test.iloc[:,2:3].values

# preparing inputs for test
full_base = pd.concat((base['Open'], base_test['Open']), axis = 0)
inputs = full_base[len(full_base) - len(base_test) - 90:].values
inputs = inputs.reshape(-1, 1)
inputs = norm.transform(inputs)

# loop for filling variable
x_test = []
for i in range (90, inputs.size):
    x_test.append(inputs[i-90:i, 0])
# format adapting
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction = regressor.predict(x_test)
# undo normalization for better viewing our results
prediction = norm.inverse_transform(prediction)

# visualization
plt.plot(real_price_open, color = 'red', label = 'Real open price')
plt.plot(real_price_high, color = 'black', label = 'Real high price')

plt.plot(prediction[:,0], color = 'blue', label = 'Open prediction')
plt.plot(prediction[:,1], color = 'green', label = 'High prediction')

plt.title('PETR4 stock price prediction')
plt.xlabel('Time (days)')
plt.ylabel('Price (R$)')
plt.legend()
plt.grid()
plt.show()
