# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 03:17:44 2025

@author: seid a
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

def data_cleaning(data,ticker):
    print(f"{ticker} Missing Values: \n {data.isnull().sum()}")
    
    # drop rows with null values
    data.dropna(inplace=True)
    return data

def Feature_engineering(data):
    returns = data.pct_change().dropna()

    return returns


def EDA(data,ticker):
    returns = data.pct_change().dropna()
    data.plot(title= f'Historical Adjust Close Prices of {ticker}')
    plt.show()
    
    returns.plot(title=f'Daily returns of {ticker}')
    plt.show()
def Analyze_volatility(data):
    rolling_volatility = data.rolling(window=30).std()*np.sqrt(252)
    return rolling_volatility

    


def rollingAvgAndStd(stockData,tickers):
    # Calculate rolling averages and standard deviations
    for data, ticker in zip(stockData,tickers):
        data['Rolling_Mean'] = data['Close'].rolling(window=30).mean()
        data['Rolling_Std'] = data['Close'].rolling(window=30).std()
        data['Rolling_Mean'].fillna(0, inplace=True) 
        data['Rolling_Std'].fillna(0, inplace=True)
        
        # Plot rolling mean and std
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.plot(data['Date'], data['Rolling_Mean'], label='30-Day Rolling Mean')
        plt.plot(data['Date'], data['Rolling_Std'], label='30-Day Rolling Std', linestyle='--')
        plt.title(f'{ticker} Volatility with Rolling Mean & Standard Deviation')
        plt.xlabel('Date')
        plt.ylabel('Price / Volatility')
        plt.legend()
        plt.xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2025-01-31"))
        plt.show()


def detect_outliers(stockData, tickers):
    """
    Detects and plots outliers in Adjusted Close Price for each stock using Z-score.

    Parameters:
        stockData (list of DataFrames): List of stock price DataFrames (each with 'Date' and 'Adj Close' columns).
        tickers (list of str): Corresponding stock ticker symbols.
    """
    for data, ticker in zip(stockData, tickers):
        data = data.copy()  # Avoid modifying the original DataFrame
        
        # Ensure Date column exists and is set as index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime if not already
            data = data.set_index('Date')

        # Compute Z-score
        data['Z-Score'] = (data['Adj Close'] - data['Adj Close'].mean()) / data['Adj Close'].std()
        outliers = data[data['Z-Score'].abs() > 3]  # Outliers: Z-score > 3 or < -3

        # Plot Adjusted Close Price with outliers
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Adj Close'], label=f'{ticker} Adjusted Close Price', color='blue', linewidth=1.5)
        plt.scatter(outliers.index, outliers['Adj Close'], color='red', label='Outliers', zorder=5)
        plt.title(f'{ticker} Outliers in Adjusted Close Price', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        # Print Outliers DataFrame in requested format
        print(f"\nOutliers for {ticker}:")
        if outliers.empty:
            print("Empty DataFrame\nColumns: [Price, Adj Close, Z-Score]\nIndex: []")
        else:
            formatted_outliers = outliers[['Adj Close', 'Z-Score']].copy()
            formatted_outliers.index.name = "Date"
            formatted_outliers.columns = ["Price", "Z-Score"]
            formatted_outliers["Ticker"] = ticker
            print(formatted_outliers)






def remove_outliers(stockData, tickers, threshold=3):
    """
    Detects and removes outliers in Adjusted Close Price using Z-score.

    Parameters:
        stockData (list of DataFrames): List of stock price DataFrames (each with 'Date' and 'Adj Close' columns).
        tickers (list of str): Corresponding stock ticker symbols.
        threshold (float): Z-score threshold for defining outliers (default is 3).

    Returns:
        list of DataFrames: Cleaned DataFrames with outliers removed.
    """
    cleaned_data = []

    for data, ticker in zip(stockData, tickers):
        data = data.copy()  # Avoid modifying the original DataFrame
        
        # Ensure 'Date' is set as index
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')

        # Compute Z-score
        data['Z-Score'] = (data['Adj Close'] - data['Adj Close'].mean()) / data['Adj Close'].std()

        # Identify outliers
        outliers = data[np.abs(data['Z-Score']) > threshold]

        # Remove outliers
        data_cleaned = data[np.abs(data['Z-Score']) <= threshold].drop(columns=['Z-Score'])

        # Store cleaned data
        cleaned_data.append(data_cleaned)

        # Print removed outliers
        print(f"\nRemoved Outliers for {ticker}:")
        if outliers.empty:
            print("No outliers found.")
        else:
            print(outliers[['Adj Close', 'Z-Score']])

    return cleaned_data



def calc_daily_return(stockData,tickers):
    for data, ticker in zip(stockData, tickers):
        data['Daily Return'] = data['Adj Close'].pct_change() * 100 
        data.dropna(inplace=True)

def plot_daily_percentage(stockData, tickers):
    for data, ticker in zip(stockData, tickers):
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Daily Return'], label=f'{ticker} Daily Returns')
        
        plt.title(f'{ticker} Daily Percentage Change')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True)

        # Format the x-axis to show readable dates
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.show()


def plot_significant_anomalies(stockData,tickers):
    threshold = 5  # 5% threshold for high/low returns

    for data, ticker in zip(stockData, tickers):
        high_returns = data[data['Daily Return'] > threshold]
        low_returns = data[data['Daily Return'] < -threshold]

        # Plot high and low returns
        plt.figure(figsize=(10, 6))
        plt.plot(data['Daily Return'],  label=f'{ticker} Daily Returns')
        plt.scatter(high_returns.index, high_returns['Daily Return'], color='green', label='High Returns', zorder=5)
        plt.scatter(low_returns.index, low_returns['Daily Return'], color='red', label='Low Returns', zorder=5)
        plt.title(f'{ticker}Days with Unusually High/Low Returns')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"{ticker}High Returns for:")
        print(high_returns[['Daily Return']])
        print("\n")
        
        print(f"{ticker}Low Returns for:")
        print(low_returns[['Daily Return']])
        print("\n")


    
def timeSeriesDecomposition(stockData,tickers):
    # Time Series Decomposition
    for data, ticker in zip(stockData,tickers):
        decomposition = seasonal_decompose(data['Close'], model='additive', period=252)
        plt.figure(figsize=(12,6))
        decomposition.plot()
        plt.suptitle(f'{ticker} Time Series Decomposition')
        plt.xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2025-01-31"))
        plt.show()



def volatility_rolling(window_size,stockData,tickers):
    # Analyze volatility for each asset
    for data, ticker in zip(stockData,tickers):
        # Calculate the rolling mean and standard deviation for the adjusted close price
        rolling_mean = data['Adj Close'].rolling(window=window_size).mean()
        rolling_std = data['Adj Close'].rolling(window=window_size).std()

        # Plot the adjusted close price along with rolling mean and rolling standard deviation
        plt.figure(figsize=(12, 8))

        # Plot the adjusted close price
        plt.subplot(311)
        plt.plot(data['Adj Close'], label=f'{ticker} Adjusted Close', color='black')
        plt.title(f'{ticker} - Adjusted Close Price')
        plt.legend()

        # Plot the rolling mean
        plt.subplot(312)
        plt.plot(rolling_mean, label=f'{ticker} {window_size}-Day Rolling Mean', color='blue')
        plt.title(f'{ticker} - {window_size}-Day Rolling Mean')
        plt.legend()

        # Plot the rolling standard deviation
        plt.subplot(313)
        plt.plot(rolling_std, label=f'{ticker} {window_size}-Day Rolling Std Dev', color='red')
        plt.title(f'{ticker} - {window_size}-Day Rolling Std Dev (Volatility)')
        plt.legend()

        plt.tight_layout()
        plt.show()



def varAndSharpeRatio(stockData, tickers):
    VaRs = {}  # Store VaR values
    Sharpe_ratios = {}  # Store Sharpe Ratios

    for data, ticker in zip(stockData, tickers):
        if 'Daily_Return' not in data.columns:
            print(f"Skipping {ticker}: 'Daily_Return' column not found.")
            continue
        
        # ✅ Calculate VaR (5th percentile)
        VaR = data['Daily_Return'].quantile(0.05)
        VaRs[ticker] = VaR
        
        # ✅ Calculate Sharpe Ratio
        mean_return = data['Daily_Return'].mean()
        std_dev_return = data['Daily_Return'].std()
        sharpe_ratio = mean_return / std_dev_return * np.sqrt(252)  # 252 trading days
        Sharpe_ratios[ticker] = sharpe_ratio

    # ✅ Create VaR Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(VaRs.keys(), VaRs.values(), color='red')
    plt.xlabel('Ticker')
    plt.ylabel('VaR (Lower is Riskier)')
    plt.title('Value at Risk (VaR) at 5% Confidence Level')
    plt.grid(axis='y')
    plt.show()

    # ✅ Create Sharpe Ratio Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(Sharpe_ratios.keys(), Sharpe_ratios.values(), color='purple')
    plt.xlabel('Ticker')
    plt.ylabel('Sharpe Ratio (Higher is Better)')
    plt.title('Sharpe Ratios of Stocks')
    plt.grid(axis='y')
    plt.show()

    # ✅ Print Values
    print("\nValue at Risk (VaR) at 5% Confidence Level:")
    for ticker, value in VaRs.items():
        print(f"{ticker}: {value:.4f}")
    
    print("\nSharpe Ratios:")
    for ticker, value in Sharpe_ratios.items():
        print(f"{ticker}: {value:.4f}")


def correlation_returns(daily_returns):
    # Calculate the correlation matrix
    corr_matrix = daily_returns.corr()

    # Plotting the correlation matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def covariance_returns(daily_returns):
    # Calculate the covariance matrix
    cov_matrix = daily_returns.cov()


    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, fmt=".8f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Covariance Matrix Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()

def daily_plot_VaR(df,var_Tesla):
    plt.figure(figsize=(10, 6))
    plt.hist(df['TSLA_daily_return'].dropna(), bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(var_Tesla, color='red', linestyle='dashed', linewidth=2, label=f'VaR (95%): {var_Tesla:.4f}')
    plt.title("Tesla's Daily Returns Distribution with VaR at 95% Confidence")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()



def daily_annual_sharpe_ratio(sharpe_ratios):
    # Plot the Sharpe ratios
    plt.figure(figsize=(8, 5))
    plt.bar(sharpe_ratios.keys(), sharpe_ratios.values(), color=['coral', 'skyblue'])
    plt.title("Comparison of Daily and Annualized Sharpe Ratios")
    plt.ylabel("Sharpe Ratio")
    plt.ylim(0, max(sharpe_ratios.values()) * 1.2)
    plt.show()



def montecarlo_simulation(df,mean_returns,cov_matrix,optimized_weights):
    dataset_size = len(df)  # Number of rows (days) in your dataset
    num_days = len(df)  # The number of data points (days) in the dataset

    num_simulations = min(1000, int(dataset_size / 10))

    simulated_portfolios = np.zeros((num_simulations, num_days))

    for i in range(num_simulations):
        # Generate random returns for each asset (TSLA, BND, SPY)
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        
        # Calculate the portfolio returns for each day (weighted sum of returns)
        simulated_portfolios[i] = np.dot(random_returns, optimized_weights)

    # Plot the simulated portfolio returns
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_portfolios.T, color='skyblue', alpha=0.1)
    plt.title('Monte Carlo Simulation: Simulated Portfolio Performance')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Daily Return')
    plt.show()

      # Calculate the cumulative return for each simulation to see the total portfolio growth over time
    cumulative_returns = np.cumsum(simulated_portfolios, axis=1)

    # Plot the cumulative returns to visualize portfolio growth over time
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.T, color='skyblue', alpha=0.1)
    plt.title('Monte Carlo Simulation: Cumulative Portfolio Return')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Portfolio Return')
    plt.show()



def cumulative_returns_indiv_assets(df, weighted_daily_return):
    cumulative_returns = (1 + weighted_daily_return).cumprod()
    cumulative_returns_TESLA = (1 + df['TSLA_daily_return']).cumprod()
    cumulative_returns_BND = (1 + df['BND_daily_return']).cumprod()
    cumulative_returns_SPY = (1 + df['SPY_daily_return']).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Optimized Portfolio", color='skyblue')
    plt.plot(cumulative_returns_TESLA, label="Tesla (TSLA)", color='gray')
    plt.plot(cumulative_returns_BND, label="Bond (BND)", color='green')
    plt.plot(cumulative_returns_SPY, label="S&P 500 (SPY)", color='orange')

    plt.title("Cumulative Returns of Portfolio and Individual Assets")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def cumulative_returns_indiv_assets(df, weighted_daily_return):
    cumulative_returns = (1 + weighted_daily_return).cumprod()
    cumulative_returns_TESLA = (1 + df['TSLA_daily_return']).cumprod()
    cumulative_returns_BND = (1 + df['BND_daily_return']).cumprod()
    cumulative_returns_SPY = (1 + df['SPY_daily_return']).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Optimized Portfolio", color='skyblue')
    plt.plot(cumulative_returns_TESLA, label="Tesla (TSLA)", color='red')
    plt.plot(cumulative_returns_BND, label="Bond (BND)", color='green')
    plt.plot(cumulative_returns_SPY, label="S&P 500 (SPY)", color='orange')

    plt.title("Cumulative Returns of Portfolio and Individual Assets")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()



def risk_return_analysis(df, mean_returns, average_portfolio_return, portfolio_volatility):
    # Plot Risk vs Return for the assets and portfolio
    returns = [mean_returns['TSLA_daily_return'], mean_returns['BND_daily_return'], mean_returns['SPY_daily_return'], average_portfolio_return]
    volatility = [df['TSLA_daily_return'].std(), df['BND_daily_return'].std(), df['SPY_daily_return'].std(), portfolio_volatility]

    plt.figure(figsize=(10, 6))

    # Scatter plot for individual assets with separate colors and labels
    plt.scatter(volatility[0], returns[0], color='skyblue', label='Tesla', s=100)
    plt.scatter(volatility[1], returns[1], color='green', label='Bond', s=100)
    plt.scatter(volatility[2], returns[2], color='orange', label='SPY', s=100)

    # Scatter plot for the optimized portfolio
    plt.scatter(portfolio_volatility, average_portfolio_return, color='blue', label="Optimized Portfolio", marker='x', s=100)

    # Add labels and legend
    for i, txt in enumerate(['TSLA', 'BND', 'SPY', 'Portfolio']):
        plt.annotate(txt, (volatility[i], returns[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Risk vs Return Analysis")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()