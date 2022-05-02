import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from pickletools import optimize
from sklearn.preprocessing import MinMaxScaler

def ticker_data(ticker = 'MSFT' , start_time = '2021-01-01'):
   
    stock = yf.Ticker(ticker)
    df = stock.history(start = start_time)
    df.to_csv('../CSV/{ticker}.csv')

    data = df.filter(['Close'])
    dataset = data.to_numpy()

    return data , dataset

def build_model(dataset, train_size, test_case_size):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler_data = scaler.fit_transform(dataset)

   
    train_set = scaler_data[0: train_size, :]

    x_train = []
    y_train = []

    for i in range(test_case_size, train_size):
        x_train.append(train_set[i-test_case_size: i, 0])
        y_train.append(train_set[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size = 1, epochs = 1)

    test_data = scaler_data[train_size -test_case_size:, :]

    x_test = []
    y_test = dataset[train_size:, :]
    for i in range(test_case_size, len(test_data)):
        x_test.append(test_data[i-test_case_size:i, 0])


    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    return y_test, x_test, scaler, model

def model_predictions(y_test, x_test, scaler, model):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean( predictions - y_test)**2)
    print(rmse)
    return predictions

def Visualize(data, predictions, train_size):
    train = data[:train_size]
    valid = data[train_size:]
    valid['Predictions'] = predictions

    #Visualize
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price USD ($)', fontsize = 18)
    plt.plot(train["Close"])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predicitons'], loc = 'lower right')
    plt.show()
    plt.savefig("../plots/visualize.png")
