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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# read data
base = pd.read_csv('petr4_train.csv')
# clean NaNs
base = base.dropna()
# select train columns
base_train = base.iloc[:, 1:7].values

# normalizing
norm = MinMaxScaler(feature_range=(0,1))
base_train_norm = norm.fit_transform(base_train)
# new normalizer for different dimensions
norm2 = MinMaxScaler(feature_range=(0,1))
norm2.fit_transform(base_train[:, 0:1])

# initialing variables
prev = []
real_price = []

# filling for 90-day prediction
for i in range(90, len(base_train_norm)):
    prev.append(base_train_norm[i-90:i, 0:6])
    real_price.append(base_train_norm[i, 0])

# adapting to keras formats
prev, real_price = np.array(prev), np.array(real_price)


# starting regressor
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (prev.shape[1], 6)))
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
regressor.add(Dense(units = 1, activation = 'sigmoid'))

# adding callbacks from Keras to improve learning
# early stopping
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)

# RLR reduce learning rate on plateau
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)

# model checkpoint
mcp = ModelCheckpoint(filepath = 'weights.h5', monitor = 'loss', save_best_only = True, verbose = 1)


# compiling using adam
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
regressor.fit(prev, real_price, epochs = 100, batch_size = 32, callbacks = [es, rlr, mcp])


# testing phase
# read test data
base_test = pd.read_csv('petr4_test.csv')
real_price_test = base_test.iloc[:,1:2].values

# preparing inputs for test
frames = [base, base_test]
full_base = pd.concat(frames)
full_base = full_base.drop('Date', axis = 1)

inputs = full_base[len(full_base) - len(base_test) - 90:].values
inputs = norm.transform(inputs)

# loop for filling variable
x_test = []
for i in range (90, len(inputs)):
    x_test.append(inputs[i-90:i, 0:6])
# format adapting
x_test = np.array(x_test)

prediction = regressor.predict(x_test)
# undo normalization for better viewing our results
prediction = norm2.inverse_transform(prediction)


# visualization
plt.plot(real_price_test, color = 'red', label = 'Real price')
plt.plot(prediction, color = 'blue', label = 'Prediction')
plt.title('PETR4 stock price prediction')
plt.xlabel('Time (days)')
plt.ylabel('Price (R$)')
plt.legend()
plt.grid()
plt.show()









