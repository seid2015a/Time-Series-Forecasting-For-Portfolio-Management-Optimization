# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 22:28:48 2025

@author: seid a.
"""
import yfinance as yf
import pandas as pd

import requests

def data_loader():
  tickers = ['TSLA','BND','SPY']
  start_date = '2015-07-01'
  end_date = '2025-07-31'
  data_frames = {}
  #session = requests.Session() 
  #session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
  # Pass the session to the download function
  for ticker in tickers:    
      data = yf.download(ticker, start=start_date, end=end_date)
      # If 'Adj Close' is missing, assume it's the same as 'Close'    
      if 'Adj Close' not in data.columns:        
          data['Adj Close'] = data['Close']
      # Ensure column order     
      data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]     
      data_frames[ticker] = data
  return data_frames["TSLA"], data_frames["BND"], data_frames["SPY"]

def format_date(data):
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data

    
