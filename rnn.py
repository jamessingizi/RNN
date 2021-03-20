#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:24:45 2021

@author: the-james
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Part 1 - Data preprocessing

#import the training set
#ranges in python have upper range/bound excluded
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # the .values makes it a numpy array

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)



#creating a data structure with 60 time steps and 1 output
X_train = []
y_train = []

#populate the lists with the 60 previous stock prices at time t

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)


#reshaping the data (adding dimensionality)
#reshape function is used anytime you want to add a dimension to a numpy array
#reshaping helps match the input tensor, (batch_size, timesteps, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Part2 - Building the RNN (LSTM)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the RNN
# regression-predicting a continous value classification-predicting single value
regressor = Sequential() 

#add the first LSTM Layer and some dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Add second LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Add third LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Add fourth LSTM layer with dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units=1))

#compiling the RNN
#mse is good for regression (continuos values in predictions)
regressor.compile(optimizer='adam', loss='mean_squared_error')

#fitting RNN to training set
regressor.fit(X_train, y_train,epochs=100,batch_size=32)

#Part 3 - Making Predictions and visualizing results
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


#getting the predicted stock price of january 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

#reshape function with -1 and 1 as arguments to avoid shape errors.
#get the right nupy shape
inputs = inputs.reshape(-1,1)
 
#scaling the inputs to match training scale
#here we dont use fit transform because the sc object was alreadt fit to the training set
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

#predicted prices are scaled, so we need to reverse the scaling to get actual stock prices

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the results
plt.plot(real_stock_price, color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





































