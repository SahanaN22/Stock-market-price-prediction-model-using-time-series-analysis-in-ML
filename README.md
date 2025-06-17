# Stock-market-price-prediction-model-using-time-series-analysis-in-ML
# 1. What was the change in price of the stock overtime?
# !pip install -q yfinance

The stock market is an integral part of any country’s financial system, influencing both the economy and individual financial prosperity. Stock price prediction is a highly attractive area for investors, traders, and researchers because accurate forecasts can lead to significant financial gains. However, stock markets are inherently dynamic and volatile, where prices are affected by a multitude of factors like political events, market sentiment, and economic indicators, making predictions extremely challenging.
Traditional statistical approaches like moving averages and ARIMA models have been used for stock forecasting for decades. However, they often fail to capture non-linear, complex patterns within the data. With the advancement in computational capabilities, the application of machine learning techniques, particularly deep learning models such as Long Short-Term Memory (LSTM) networks and intelligent time series models like Facebook Prophet, have opened new avenues for more accurate stock price forecasting.
In this project, we develop a hybrid predictive model integrating ARIMA, LSTM, and Prophet to capture both the linear and non-linear dependencies in stock market data. Our system focuses on providing investors with reliable, interpretable, and easy-to-use forecasts, thereby aiding in making data-driven investment decisions and managing financial risks.

Problem Statement
This project focuses on designing and developing a machine learning-based stock price prediction system using Time Series Analysis. The system aims to address the limitations of existing stock forecasting methods by integrating ARIMA, LSTM, and Facebook Prophet models to enhance prediction accuracy.
Specifically, the objectives of the problem statement are:
To accurately predict future stock prices based on historical data.
To capture both linear trends and complex non-linear patterns.
To handle trend changes, seasonality, and missing values effectively.
To present the predictions through an intuitive web interface for easy access by users.
The final system is expected to assist investors and traders in making informed, strategic financial decisions while reducing reliance on speculation and manual forecasting methods.

Scope
The scope of this project encompasses the collection of historical stock data, preprocessing the data for modeling, training multiple predictive models (ARIMA, LSTM, Prophet), evaluating model performances, and deploying the best-performing models into a web-based interface. The platform is designed to predict daily closing stock prices for major listed companies.
The project is restricted to time series forecasting based on historical prices (technical analysis) and does not consider real-time sentiment analysis, news impact, or macroeconomic indicators. However, the modularity of the system allows easy integration of such features in the future.

Objectives
The key objectives of the project are:
To collect and preprocess historical stock market data.
To develop forecasting models using ARIMA for linear patterns, LSTM for non-linear sequence modeling, and Facebook Prophet for trend/seasonality analysis.
To evaluate the performance of each model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
To compare the models and determine the best forecasting strategy.
To design a Streamlit-based web application that allows users to interact with the prediction system easily.

## Descriptive Statistics about the Data
`.describe()` generates descriptive statistics. Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset’s distribution, excluding `NaN` values.

Analyzes both numeric and object series, as well as `DataFrame` column sets of mixed data types. The output will vary depending on what is provided. Refer to the notes below for more detail.

# Summary Stats
GOOG.describe()

