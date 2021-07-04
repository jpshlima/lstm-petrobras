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


