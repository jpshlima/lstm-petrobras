# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:07:32 2021

@author: jpshlima
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# read data
base = pd.read_csv('petr4_train.csv')
# clean NaNs
base = base.dropna()

# select our targets
base_train = base.iloc[:, 1:2].values

# normalizing
norm = MinMaxScaler(feature_range=(0,1))
base_train_norm = norm.fit_transform(base_train)

# initialing variables
prev = []
real_price = []

# filling for 90-day prediction
for i in range(90, base_train_norm.size):
    prev.append(base_train_norm[i-90:i, 0])
    real_price.append(base_train_norm[i, 0])

# adapting formats (only 1 dimension, i.e., using opening price only)
prev, real_price = np.array(prev), np.array(real_price)
prev = np.reshape(prev, (prev.shape[0], prev.shape[1], 1))

