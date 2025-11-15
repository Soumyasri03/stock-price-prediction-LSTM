
# Stock Trend Prediction Using LSTM

This project is a Streamlit-based web application that predicts stock price trends using an LSTM deep learning model. It also provides technical indicators, visualization tools, and future price forecasting. The purpose of this application is to help users understand stock movements using a combination of machine learning and technical analysis.

## Features

### 1. Stock Data Fetching
- Fetches historical market data for any stock ticker using Yahoo Finance.
- Supports NSE and BSE tickers (for example: KOTAKBANK.NS, RELIANCE.NS).
- Allows users to choose custom date ranges for analysis.

### 2. Data Visualization
- Interactive line charts using Plotly.
- Traditional closing price charts using Matplotlib.
- Moving Average charts:
  - 100-day Moving Average
  - 200-day Moving Average
  - Combined MA chart for trend comparison

### 3. LSTM-Based Price Prediction
- Preprocessing of data using MinMax scaling.
- Data split into 80% training and 20% testing.
- Uses a pretrained LSTM model saved as `keras_model.h5`.
- Visual comparison of actual vs predicted closing prices.
- Ability to forecast user-defined number of future days.

### 4. Technical Indicators Included
- Accumulation/Distribution Oscillator (ADO)
- MACD (Moving Average Convergence Divergence)
- Weighted Moving Average (WMA)
- Exponential Moving Averages (EMA50, EMA100, EMA200)
- Bollinger Bands
- Combined technical indicator visualizations including candlesticks

### 5. Future Price Forecasting
- Predicts upcoming stock prices based on most recent trends.
- Displays future price movements using both Matplotlib and Plotly charts.
- Users can select how many days they want to predict.

### 6. Additional Analysis Tools
- Daily percentage change calculation
- Annual return estimation based on percentage change

## Project Structure

```
.
├── stock.py               # Main Streamlit application file
├── keras_model.h5         # Pretrained LSTM model file
├── LSTM.ipynb             # Notebook used to train the LSTM model
└── README.md              # Documentation file
```



## How the LSTM Model Works

1. Fetches historical closing prices from Yahoo Finance.  
2. Normalizes values using MinMaxScaler.  
3. Uses the previous 100 timesteps to predict the next closing price.  
4. Loads the pretrained LSTM model to make predictions.  
5. Converts predicted values back to original price range.  
6. Allows forecasting of future days based on model predictions.

## Technical Indicators Overview

### Accumulation/Distribution Oscillator (ADO)
Measures buying and selling pressure based on volume and price.

### MACD
Momentum indicator calculated using:
- 12-day EMA  
- 26-day EMA  
- 9-day Signal line  
Includes histogram visualization.

### Weighted Moving Average (WMA)
Gives more importance to recent data during averaging.

### Exponential Moving Averages (50, 100, 200)
Tracks short-term and long-term trends.

### Bollinger Bands
Shows volatility using upper, lower and middle bands.


## Future Improvements

- Add indicators such as RSI, Stochastic Oscillator, and VWAP.
- Allow model retraining directly within the application.
- Add alerts or notifications for price breakouts.
- Include multiple feature inputs such as high, low, and volume.


