# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:49:23 2021

@author: SUSMITA DAS
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_sample =[]
train_label=[]

for i in range(1000):
    younger_ages=randint(13, 60)
    train_sample.append(younger_ages)
    train_label.append(0)
    
    older_ages=randint(60, 100)
    train_sample.append(older_ages)
    train_label.append(1)

train_sample = np.array(train_sample)
train_label = np.array(train_label)

scalar = MinMaxScaler(feature_range=(0,1))
scalar_train_sample = scalar.fit_transform(train_sample.reshape(-1,1))

model = Sequential([Dense(16, input_dim=1, activation = 'relu'), Dense(32, activation='relu'), Dense(2, activation='softmax')])
model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_sample, train_label, batch_size=10, epochs=10)


### Creating test sample
test_sample =[]
test_label=[]

for i in range(500):
    younger_ages=randint(13, 60)
    test_sample.append(younger_ages)
    test_label.append(0)
    
    older_ages=randint(60, 100)
    test_sample.append(older_ages)
    test_label.append(1)
    
test_sample = np.array(test_sample)
test_label = np.array(test_label)

test_sample_output=model.predict_classes(test_sample, batch_size=10)

from sklearn.metrics import confusion_matrix
predicted_values=confusion_matrix(test_label,test_sample_output )
