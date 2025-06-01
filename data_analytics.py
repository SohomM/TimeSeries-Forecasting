import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

data = yf.download(['AAPL'], start='2015-01-01', end='2024-05-12')
print(data.head())
data.to_csv('AAPL_yahoo.csv')

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' directory")

# Use the downloaded data directly instead of trying to read a non-existent file
# Convert the multi-index DataFrame to a single index
if isinstance(data.columns, pd.MultiIndex):
    # If it's a multi-index DataFrame (common with yfinance)
    df = data.copy()
    # No need to set index as it's already datetime
else:
    # If it's a single index DataFrame
    df = data.copy()

print("Using downloaded data directly")
print(df.shape)  # (number of rows, number of columns)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Forward fill then backward fill gaps
df_clean = df.ffill().bfill()
print("Missing values after cleaning:", df_clean.isnull().sum().sum())
z_scores = stats.zscore(df_clean['Close'])
df_clean = df_clean[(np.abs(z_scores) < 3)]  # Keep values within 3σ

# Moving Averages
df_clean['MA_5'] = df_clean['Close'].rolling(window=5).mean()
df_clean['MA_21'] = df_clean['Close'].rolling(window=21).mean()
# Daily Returns & Volatility
df_clean['Daily_Return'] = df_clean['Close'].pct_change()
df_clean['Volatility'] = df_clean['Daily_Return'].rolling(window=21).std() * np.sqrt(21)
# Relative Strength Index (RSI)
delta = df_clean['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df_clean['RSI'] = 100 - (100 / (1 + rs))

plt.figure(figsize=(14,7))
plt.plot(df_clean['Close'], label='Close Price', alpha=0.5)
plt.plot(df_clean['MA_5'], label='5-Day MA', color='orange')
plt.plot(df_clean['MA_21'], label='21-Day MA', color='purple')
plt.title('AAPL Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

fig = go.Figure(data=[go.Candlestick(x=df_clean.index,
                open=df_clean['Open'],
                high=df_clean['High'],
                low=df_clean['Low'],
                close=df_clean['Close'])])
fig.add_trace(go.Bar(x=df_clean.index,
                    y=df_clean['Volume'],
                    name='Volume',
                    marker_color='rgba(100, 102, 200, 0.4)',
                    yaxis='y2'))
fig.update_layout(
    title='AAPL Candlestick Chart with Volume',
    yaxis_title='Price ($)',
    yaxis2=dict(title='Volume', overlaying='y', side='right'),
    xaxis_rangeslider_visible=False
)
fig.show()
# Save to the data directory we created earlier
df_clean.to_csv('data/AAPL_cleaned.csv')
print("Saved cleaned data to 'data/AAPL_cleaned.csv'")

result = adfuller(df_clean['Close'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
# Ensure index is DatetimeIndex and set frequency to business days
df_clean.index = pd.to_datetime(df_clean.index)
df_clean = df_clean.asfreq('B')

# If p-value > 0.05, data is non-stationary; apply differencing.
df_stationary = df_clean['Close'].diff().dropna()
# model = ARIMA(df_clean['Close'], seasonal=False, trace=True)
# print(model.summary())
model = ARIMA(df_clean['Close'], order=(1,1,1))  # Example: p=1, d=1, q=1
model_fit = model.fit()
# Forecast
arima_forecast = model_fit.forecast(steps=10)

sarima_model = SARIMAX(df_clean['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima_model.fit()
print(sarima_result.summary())
# For ARIMA from statsmodels, use forecast() not predict()
arima_forecast_30 = model_fit.forecast(steps=30)
# For SARIMA:
sarima_forecast = sarima_result.forecast(steps=30)

plt.figure(figsize=(12,6))
plt.plot(df_clean['Close'], label='Observed')
plt.plot(pd.date_range(df_clean.index[-1], periods=31, freq='B')[1:], arima_forecast_30, label='ARIMA Forecast')
plt.legend()
plt.show()

# Prepare DataFrame for Prophet
df_clean = df_clean.copy()
df_clean = df_clean.reset_index()
df_clean.rename(columns={df_clean.columns[0]: 'ds', 'Close': 'y'}, inplace=True)
prophet_df = df_clean[['ds', 'y']]
m = Prophet()
m.fit(prophet_df)
future = m.make_future_dataframe(periods=30)
prophet_forecast = m.predict(future)
fig1 = m.plot(prophet_forecast)
fig2 = m.plot_components(prophet_forecast)
plt.show()

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df_clean['Close'].values.reshape(-1,1))
# Create sequences
X, y = [], []
for i in range(60, len(scaled_close)):
    X.append(scaled_close[i-60:i])
    y.append(scaled_close[i])
X, y = np.array(X), np.array(y)
# Build LSTM
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X, y, epochs=20, batch_size=32)

# Add prediction code for LSTM model
lstm_pred = lstm_model.predict(X)
print("LSTM model training complete")
