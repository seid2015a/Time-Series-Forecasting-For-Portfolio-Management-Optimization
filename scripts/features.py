# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 05:09:50 2025

@author: seid a
"""


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sco
import scipy.stats as stats

def plot_data(data,title):
    plt.figure(figsize = (12,6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Price')
    plt.show()
    
def split_data(data):
    train_size = int(len(data)*0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data,test_data
def fit_arima(train):
    try:
        train_model = auto_arima(train, seasonal = False, suppres_warnings=True, stepwise=True)
        return train_model
    except Exception as e:
        print(f"Error: fitting ARIMA model: {e}")
        return None

def fit_sarima(train,order,seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    return model.fit()

def prepare_lstm_data(data,time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:(i+time_step),0])
        Y.append(data[i + time_step,0])
    return np.array(X), np.array(Y)
def build_and_train_lstm(X_train, y_train, epochs = 10, batch_size = 32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size)
    return model

def calculate_metrics(actual, predicted, epsilon=1e-10):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)  # Replace near-zero values with epsilon
    
    mape = np.mean(np.abs((actual_safe - predicted) / actual_safe)) * 100  # Avoid division by zero
    
    return mae, rmse, mape


def plot_metrics(metrics, title):
    model_names = ['ARIMA', 'SARIMA', 'LSTM']
    mae = [metrics['ARIMA'][0], metrics['SARIMA'][0], metrics['LSTM'][0]]
    rmse = [metrics['ARIMA'][1], metrics['SARIMA'][1], metrics['LSTM'][1]]
    mape = [metrics['ARIMA'][2], metrics['SARIMA'][2], metrics['LSTM'][2]]

    x = np.arange(len(model_names))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, mae, width, label='MAE')
    ax.bar(x, rmse, width, label='RMSE')
    ax.bar(x + width, mape, width, label='MAPE')

    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.show()


def run_forecasting(stockData, asset_name,seasonal_order=(1, 1, 1, 12), forecast_days=360):
    print(f"Running forecasting for {asset_name}...")
    
    plot_data(stockData, f'{asset_name} Stock Prices')

    # Split data
    train, test = split_data(stockData)

    #scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
    scaled_test = scaler.transform(test.values.reshape(-1, 1))


    arima_model = fit_arima(train)
    arima_forecast = arima_model.predict(n_periods=forecast_days)

    order = arima_model.order
    sarima_fit = fit_sarima(train, order=order, seasonal_order=seasonal_order)
    sarima_forecast = sarima_fit.forecast(steps=forecast_days)

    X_train, y_train = prepare_lstm_data(scaled_train, time_step=60)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    lstm_model = build_and_train_lstm(X_train, y_train)

    inputs = np.concatenate((scaled_train[-60:], scaled_test[:forecast_days]))
    X_test, y_test = prepare_lstm_data(inputs, time_step=60)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    lstm_forecast = lstm_model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast)

    arima_forecast = np.ravel(arima_forecast[:forecast_days])  
    sarima_forecast = np.ravel(sarima_forecast[:forecast_days]) 
    lstm_forecast = np.ravel(lstm_forecast[:forecast_days])

    arima_metrics = calculate_metrics(test.values[:forecast_days], arima_forecast)
    sarima_metrics = calculate_metrics(test.values[:forecast_days], sarima_forecast)
    lstm_metrics = calculate_metrics(test.values[:forecast_days], lstm_forecast)

    print(f"{asset_name} - ARIMA - MAE: {arima_metrics[0]}, RMSE: {arima_metrics[1]}, MAPE: {arima_metrics[2]}")
    print(f"{asset_name} - SARIMA - MAE: {sarima_metrics[0]}, RMSE: {sarima_metrics[1]}, MAPE: {sarima_metrics[2]}")
    print(f"{asset_name} - LSTM - MAE: {lstm_metrics[0]}, RMSE: {lstm_metrics[1]}, MAPE: {lstm_metrics[2]}")

    metrics = {
        'ARIMA': arima_metrics,
        'SARIMA': sarima_metrics,
        'LSTM': lstm_metrics
    }
    
    plot_metrics(metrics, f'Model Performance Metrics for {asset_name}')

    results = {
        'arima_forecast': arima_forecast,
        'sarima_forecast': sarima_forecast,
        'lstm_forecast': lstm_forecast,
        'test_data': test.values[:forecast_days],
        'metrics': {
            'ARIMA': arima_metrics,
            'SARIMA': sarima_metrics,
            'LSTM': lstm_metrics
        },
        'models': {
            'ARIMA': arima_model,
            'SARIMA': sarima_fit,
            'LSTM': lstm_model
        }
    }

    return results


        