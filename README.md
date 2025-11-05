# Stock Market Trend Prediction with LSTM

An intelligent stock trend analysis and forecasting system that uses **Long Short-Term Memory (LSTM)** neural networks and technical indicators to visualize, predict, and analyze stock price movements.

This application provides an interactive dashboard built with **Streamlit**, where users can enter any stock ticker, view historical data, analyze technical indicators, and generate stock price predictions for future dates.

---

## Key Features

### Stock Data Exploration
- Fetches real-time stock data using Yahoo Finance (`yfinance`).
- Allows users to select stock tickers and custom date ranges.
- Displays historical Open, High, Low, Close, and Volume data in an interactive table.

### LSTM-Based Price Prediction
- Uses trained LSTM models to forecast stock price trends.
- Predicts next-day stock movement and generates future trend lines.
- Supports user-defined number of future days to forecast.

### Visual Stock Insights
- **Time Series Graphs:** Close Price vs Time.
- **Moving Averages:** 100-day and 200-day SMA plotted with price.
- **Advanced Moving Averages:** 50-, 100-, 200-day Exponential Moving Averages (EMA).
- **Bollinger Bands:** Volatility visualized with Upper, Lower, and SMA bands.

### Technical Indicators Included

| Indicator | Purpose |
|-----------|---------|
| **MACD** | Detects trend direction and momentum. |
| **ADO** | Measures money flow using price and volume. |
| **WMA** | Emphasizes recent price movements. |
| **EMA (50, 100, 200)** | Tracks short, mid, and long-term trends. |
| **Bollinger Bands** | Visualizes volatility around moving averages. |
| **Annual Return** | Calculates yearly return based on daily percentage change. |

---

## How the Model Works

### 1. Data Collection
- Stock data is fetched with `yfinance`.
- User selects date range and ticker (e.g., `AAPL`, `TSLA`, `KOTAKBANK.NS`).

### 2. Data Preprocessing
- Only the Close price is used.
- Data is split: 80% training, 20% testing.
- MinMaxScaler normalizes values between 0 and 1.
- Uses 100 previous time steps to predict the next step.

### 3. Model Training
- LSTM model is trained in `LSTM.ipynb`.
- Saved as `keras_model.h5` for reuse.

### 4. Prediction and Forecasting
- Predicts closing prices for test data.
- Visualizes actual vs predicted stock prices.
- Forecasts future stock values based on user input.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | TensorFlow / Keras (LSTM) |
| Data Source | Yahoo Finance (`yfinance`) |
| Visualization | Plotly, Matplotlib |
| Data Handling | Pandas, NumPy |
| Scaling | Scikit-learn |

---

## Project Structure

```
project/
├── stock.py               # Main Streamlit app
├── LSTM.ipynb             # LSTM model training script
├── keras_model.h5         # Saved trained model
├── requirements.txt       # Dependencies

