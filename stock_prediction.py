#Time-series-components: trends, seasonality, irregularities, cycles, variaiton 

#Trend: overall long-term direction of the series.
#Seasonality: When their are repeated intervals in the data and is related to season behavour.
#cycles: Up and down cycle that is not seasonal like sin wave.
#Variation: Rando varition or not random varition.
#Irregularities: they may be jumps or drop that are one off events.

import math
from pickletools import optimize
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.model import sequential 
from keras.layers import Dense, LSTM

msft = yf.Ticker("MSFT")
df = msft.history(start ='2021-01-01')
df.to_csv('msft.csv')

df = pd.read_csv('msft.csv')

#plt.plot(df['Date'] , df['Close'])
#plt.show()

#data set for training
data = df.filter(['Close'])
dataset = data.values

#scales data into 1-0
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

#training sets 
train_size = int(df.shape[0] * 0.8)
train_set = scaler_data[0: train_size, :]

x_train = []
y_train = []

for i in range(60, train_size):
    x_train.append(train_set[i-60: i, 0])
    y_train.append(train_set[i,0])



x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#Building the LSTM model
model = sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_')

#Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)


########################################################
#creating testing data set
test_data = scaler_data[train_size -60:, :]

x_test = []
y_test = dataset[train_size:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#convert data into numpy array
x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_tranform(predictions)

#Get the root mean square error(RMSE)
rmse = np.sqrt(np.mean( predictions - y_test)**2)
