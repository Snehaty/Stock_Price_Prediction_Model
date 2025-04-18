#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf


# In[3]:


import yfinance as yf
import pandas as pd

# 🟢 Replace 'AAPL' with your stock symbol and adjust the date range as needed
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-12-31'

# ⬇️ Download data from Yahoo Finance
stock_data = yf.download(ticker, start=start_date, end=end_date)

# 🧹 Clean up and check
stock_data.dropna(inplace=True)
print(stock_data.head())
print("✅ Data shape:", stock_data.shape)


# In[4]:


ticker_symbol='AAPL'
start_date='2020-2-20'
end_date='2024-12-24'
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)


# In[5]:


print(stock_data.head())


# In[6]:


stock_data.to_csv(f'{ticker_symbol}_stock_data.csv')


# In[7]:


print(stock_data.info())

print(stock_data.isnull().sum())


# In[8]:


print(stock_data.describe())


# In[9]:


import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Close Price')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# In[10]:


# Plot histogram of closing prices
plt.figure(figsize=(10, 6))
stock_data['Close'].hist(bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Closing Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()


# In[11]:


stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()


plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.plot(stock_data['SMA_50'], label='50-Day SMA', alpha=0.7)
plt.plot(stock_data['SMA_200'], label='200-Day SMA', alpha=0.7)
plt.title('Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# In[12]:


stock_data['Daily_Return'] = stock_data['Close'].pct_change()


plt.figure(figsize=(12, 6))
plt.plot(stock_data['Daily_Return'], label='Daily Return')
plt.title('Daily Returns of Stock')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.show()


print(stock_data['Daily_Return'].describe())


# In[13]:


correlation_matrix = stock_data.corr()


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose


decomposition = seasonal_decompose(stock_data['Close'], model='additive', period=252)


plt.figure(figsize=(8, 8))
decomposition.plot()
plt.show()


# In[15]:


from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(stock_data['Close'].dropna())
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')




# In[16]:


stock_data['lag_1'] = stock_data['Close'].shift(1)
stock_data['lag_2'] = stock_data['Close'].shift(2)


stock_data['rolling_mean_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['rolling_std_5'] = stock_data['Close'].rolling(window=5).std()


stock_data['daily_return'] = stock_data['Close'].pct_change()


stock_data['day_of_week'] = stock_data.index.dayofweek
stock_data['month'] = stock_data.index.month


# In[17]:


X = stock_data.drop(['Close'], axis=1)
y = stock_data['Close']


# In[18]:


print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)


train_size = int(len(X) * 0.8)


X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[19]:


plt.figure(figsize=(15,5))
plt.plot(y, label='Target (Close Price)', color='blue')
plt.axvline(x=y.index[train_size], color='red', linestyle='--', label='Train/Test Split')
plt.title("Train/Test Split Visualization")
plt.legend()
plt.show()


# In[20]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


import pandas as pd

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)


# In[22]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[23]:


data = stock_data[['Close']]


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)


# In[24]:


train_size = int(0.8 * len(X))

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[25]:


print(X_train.shape)


# In[26]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')


# In[27]:


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


# In[28]:


predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test)


# In[29]:


plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[30]:


y_pred = model.predict(X_test)


y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)


# In[31]:


from sklearn.metrics import mean_squared_error, mean_absolute_error # Import necessary functions

y_pred = model.predict(X_test)


y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test_actual, y_pred) # Now mean_squared_error is defined
mae = mean_absolute_error(y_test_actual, y_pred)
print("MSE:", mse)
print("MAE:", mae)


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


errors = y_test_actual - y_pred


plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(errors, color='orange')
plt.axhline(0, linestyle='--', color='gray')
plt.title('Prediction Errors Over Time')
plt.xlabel('Time')
plt.ylabel('Error')


plt.subplot(1, 3, 2)
sns.histplot(errors, bins=30, kde=True, color='purple')
plt.title('Error Distribution')
plt.xlabel('Error')


plt.subplot(1, 3, 3)
plt.scatter(y_test_actual, y_pred, alpha=0.5, color='green')
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)


y_pred = model.predict(X_test)




plt.figure(figsize=(10, 4))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.title('Stock Price Prediction - Improved LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")


# In[34]:


# Predict on X_test
y_pred_scaled = model.predict(X_test)

# Reshape if necessary
y_test_reshaped = y_test.reshape(-1, 1)

# Inverse transform both predictions and actual values
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test_reshaped)


# In[35]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"📊 MAE (Mean Absolute Error): {mae:.2f}")
print(f"📉 MSE (Mean Squared Error): {mse:.2f}")


# In[36]:


# Optional: Add dates if available
# df['Date'][start_index:] (depending on how you split)

results = pd.DataFrame({
    'Actual Price': y_true.flatten(),
    'Predicted Price': y_pred.flatten()
})

results.to_csv('lstm_predictions.csv', index=False)
print("✅ Predictions saved to lstm_predictions.csv")


# In[37]:


# Calculate error (residuals)
errors = y_true.flatten() - y_pred.flatten()

# Line plot of residuals
plt.figure(figsize=(12, 5))
plt.plot(errors, color='purple')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Residuals (Actual - Predicted)")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# Histogram of error distribution
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[38]:


from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Assuming 'errors' is a NumPy array or list of residuals (actual - predicted)
plt.figure(figsize=(10, 5))
plot_acf(errors, lags=40)
plt.title("Autocorrelation of Residuals")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.tight_layout()
plt.show()


# In[39]:


print("📊 Rows before feature engineering:", len(stock_data))
stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
stock_data['Momentum'] = stock_data['Close'] - stock_data['Close'].shift(10)
stock_data['Volatility'] = stock_data['Close'].rolling(window=10).std()
stock_data['RSI_14'] = 100 - (100 / (1 + stock_data['Close'].pct_change().rolling(window=14).mean() /
                                      stock_data['Close'].pct_change().rolling(window=14).std()))

# Drop rows with NaNs created by the rolling functions
stock_data.dropna(inplace=True)
print("✅ Rows after feature engineering:", len(stock_data))


# In[40]:


import yfinance as yf
stock_data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
print(stock_data.head())
print(f"✅ Rows pulled: {len(stock_data)}")


# In[41]:


# Just to be safe, start clean
stock_data = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
print(f"✅ Rows pulled: {len(stock_data)}")

# 🚫 Skip inverse scaling if not needed
# Only do inverse_transform if you had previously scaled and saved the scaler

# ✅ Feature engineering on real prices
stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
stock_data['Momentum'] = stock_data['Close'] - stock_data['Close'].shift(10)
stock_data['Volatility'] = stock_data['Close'].rolling(window=10).std()

# RSI
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
stock_data['RSI_14'] = 100 - (100 / (1 + rs))

# Drop NaNs
stock_data.dropna(inplace=True)
print("✅ Rows remaining after feature engineering:", len(stock_data))


# In[42]:


# 🚨 Your DataFrame has MultiIndex columns with blanks – clean it up
stock_data.columns = [col if isinstance(col, str) and col != '' else f"Feature_{i}"
                      for i, col in enumerate(stock_data.columns)]

print("✅ Cleaned column names:")
print(stock_data.columns)


# In[43]:


# 🔄 Assign real names to columns
stock_data.columns = [
    'Open', 'High', 'Low', 'Close', 'Adj Close',  # ← from yfinance
    'SMA_10', 'SMA_50', 'EMA_10', 'Momentum', 'Volatility', 'RSI_14'  # ← engineered
]

print("✅ Columns renamed:")
print(stock_data.columns)


# In[44]:


print(X_train.shape)
# Should show: (samples, 60, 6)


# In[45]:


features_used = ['Close']  # or 'Price' if that's what your column is called


# In[46]:


print(stock_data.columns)


# In[47]:


# Flatten MultiIndex columns
rolling_data = stock_data
rolling_data.columns = ['_'.join(filter(None, col)).strip() for col in rolling_data.columns.values]

print("✅ Flattened columns:\n", rolling_data.columns)


# In[48]:


# Predictions
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.concatenate([predicted,
                            np.zeros((len(predicted), len(features_used) - 1))], axis=1))[:, 0]

actual_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1),
                           np.zeros((len(y_test), len(features_used) - 1))], axis=1))[:, 0]

# Plot
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.title("LSTM Prediction (Multi-Feature)")
plt.show()


# In[49]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Assuming these are the inverse-scaled arrays of predictions
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE: {mae:.4f}")
print(f"✅ R² Score: {r2:.4f}")


# In[50]:


# Get the min and scale for the target feature (e.g., 'Close_AAPL' or 'Close')
target_index = features_used.index('Close')  # or 'Close' depending on your setup
min_val = scaler.data_min_[target_index]
scale_val = scaler.data_range_[target_index]

# Assuming predicted_future is the output from your model prediction
# Replace this with the actual way you get your predictions
predicted_future = model.predict(X_test)  # Or however you get future predictions

# Reverse MinMax scaling manually
predicted_future = np.array(predicted_future)
predicted_future_unscaled = predicted_future * scale_val + min_val


# In[51]:


# First, let's clean up the column names by removing the extra underscores
stock_data.columns = [col.replace('_', '') for col in stock_data.columns]

# Now check the cleaned column names
print("Cleaned columns:", stock_data.columns.tolist())

# Define our technical indicator calculation function
def calculate_technical_indicators(df):
    """Calculate technical indicators using the closing price column"""
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Momentum (price change over 5 periods)
    df['Momentum'] = df['Close'].pct_change(periods=5)

    # Volatility (rolling standard deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values from indicator calculations
    df.dropna(inplace=True)
    return df

# Calculate indicators
try:
    stock_data = calculate_technical_indicators(stock_data)
except KeyError as e:
    print(f"Error: {e}")
    print("Please verify your DataFrame contains a 'Close' column after cleaning")
    print("Current columns:", stock_data.columns.tolist())
    raise

# Now proceed with the rest of your code
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features = stock_data[['SMA_10', 'SMA_50', 'EMA_10', 'Momentum', 'Volatility', 'RSI_14']]
target = stock_data[['Close']]

scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target)

# Combine for sequence creation
model_data = np.column_stack((scaled_target.flatten(), scaled_features))

# Split chronologically
train_size = int(len(model_data) * 0.8)
train_data = model_data[:train_size]
test_data = model_data[train_size:]

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, :])
        y.append(data[i, 0])  # Target is Close price
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate model
predictions = model.predict(X_test)
predictions = target_scaler.inverse_transform(predictions)
actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f"MSE: {mse}, MAE: {mae}")

# Visualize results
plt.figure(figsize=(12,6))
plt.plot(actuals, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()


# In[52]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(actual_prices, label="Actual Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.title("LSTM Prediction (Multi-Feature)")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()


# In[53]:


# Assuming 'stock_data' contains your main stock price data
# and 'calculate_technical_indicators' function is defined
stock_data = calculate_technical_indicators(stock_data)  # Recalculate indicators
final_df = stock_data.copy()  # Make a copy called final_df

# Now proceed to use final_df in the rest of your code:
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_df[features_used])  # This should work now


# In[54]:


# Before predictions, verify:
features_used = ['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Momentum', 'Volatility', 'RSI_14']  # Define ALL features
assert model.input_shape[2] == len(features_used), \
    f"Feature count mismatch: model expects {model.input_shape[2]}, got {len(features_used)}"


# In[55]:


# Ensure your input data has exactly sequence_length timesteps
sequence_length = 60  # Define sequence_length here
current_sequence = scaled_data[-sequence_length:]  # shape should be (60, 7)
print(current_sequence.shape)  # This will print (60, 7) if correct


# In[56]:


# Assuming `current_sequence` holds the data for prediction
# Assuming 'final_df' contains your DataFrame with all features
# and 'features_used' is a list of the feature names

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_df[features_used])

# Extract the current sequence for prediction
sequence_length = 60
current_sequence = scaled_data[-sequence_length:]

# Reshape the sequence for the LSTM model
# current_sequence should now have shape (1, sequence_length, num_features)
input_sequence = current_sequence.reshape(1, sequence_length, len(features_used))

# Now you can make predictions
predicted_close = model.predict(input_sequence)[0][0]


input_sequence = current_sequence

# Since model outputs 1 value (Close price prediction)
predicted_close = model.predict(input_sequence.reshape(1, sequence_length, len(features_used)))[0][0]


# In[57]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Verify model input/output shapes
print("Model input shape:", model.input_shape)  # Should be (None, 60, 7)
print("Model output shape:", model.output_shape)  # Should be (None, 1)

# 2. Define ALL 7 features (must match training data)
features_used = ['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Momentum', 'RSI_14', 'Volatility']  # Updated to 7 features

# 3. Verify feature count matches model
if model.input_shape[2] != len(features_used):
    raise ValueError(f"Model expects {model.input_shape[2]} features but got {len(features_used)}")

# 4. Prepare data with ALL features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_df[features_used])  # Use all 7 features

# Rest of the prediction code remains the same...
sequence_length = 60
current_sequence = scaled_data[-sequence_length:]  # Shape (60, 7)

# 🟢 Define the number of future days to predict
future_days = 30  # For example, predict for the next 30 days

predicted_future = []

for _ in range(future_days):
    X_input = current_sequence.reshape(1, sequence_length, len(features_used))

    try:
        next_pred = model.predict(X_input, verbose=0)
        next_close = next_pred[0][0]  # Assuming model outputs only Close price

        # If model outputs all features, use next_pred[0][0] for Close
        # If not, you may need inverse scaling logic

        predicted_future.append(next_close)

        # Update sequence (drop oldest, append new prediction)
        # Note: Since we don't have true future features, we reuse the last values
        # This is a limitation for multi-feature prediction
        new_row = current_sequence[-1].copy()  # Copy last row
        new_row[0] = next_pred[0][0]  # Update Close price (assuming it's first feature)
        current_sequence = np.vstack([current_sequence[1:], new_row])

    except Exception as e:
        print(f"Prediction failed: {e}")
        break

# Generate future dates and plot (same as before)
future_dates = pd.date_range(
    start=final_df.index[-1] + pd.Timedelta(days=1),
    periods=future_days,
    freq='B'
)

future_df = pd.DataFrame({'Predicted_Close': predicted_future}, index=future_dates)

# Plotting code...


# In[58]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Confirm Model Shape ---
print("Model input shape:", model.input_shape)  # Should be (None, 60, 7)
print("Model output shape:", model.output_shape)  # Should be (None, 1)

# --- 2. Define Features ---
features_used = ['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'Momentum', 'RSI_14', 'Volatility']

# --- 3. Calculate Missing Features (if needed) ---
if 'RSI_14' not in final_df.columns:
    final_df['RSI_14'] = compute_rsi(final_df)  # Use RSI function from earlier
if 'Volatility' not in final_df.columns:
    final_df['Volatility'] = compute_volatility(final_df)  # Use Volatility function

# --- 4. Scale Data ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_df[features_used])

# --- 5. Initialize Prediction ---
sequence_length = 30
current_sequence = scaled_data[-sequence_length:].copy()  # Shape (60, 7)
close_index = features_used.index('Close')

# --- 6. Predict Future Prices ---
predicted_future_scaled = []
for _ in range(30):  # Predict next 30 days
    X_input = current_sequence.reshape(1, sequence_length, len(features_used))
    next_close_scaled = model.predict(X_input, verbose=0)[0][0]  # Assumes model outputs (1, 1)
    predicted_future_scaled.append(next_close_scaled)

    # Update sequence (replace Close, keep other features)
    new_row = current_sequence[-1].copy()
    new_row[close_index] = next_close_scaled
    current_sequence = np.vstack([current_sequence[1:], new_row])

# --- 7. Inverse Transform Predictions ---
close_scaler = MinMaxScaler()
close_scaler.fit(final_df[['Close']])  # Fit only on Close
predicted_future_real = close_scaler.inverse_transform(np.array(predicted_future_scaled).reshape(-1, 1))

# --- 8. Create Future Dates ---
future_dates = pd.date_range(
    start=final_df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq='B'
)

# --- 9. Plot Results ---
future_df = pd.DataFrame({'Predicted_Close': predicted_future_real.flatten()}, index=future_dates)

plt.figure(figsize=(14, 6))
plt.plot(final_df['Close'].iloc[-100:], label='Historical Prices')
plt.plot(future_df['Predicted_Close'], 'r--', label='Predicted Prices')
plt.title('60-Day Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()


# In[59]:


import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
plt.plot(final_df['Close'].iloc[-90:], label='Historical Prices', linewidth=2)
plt.plot(future_df['Predicted_Close'], 'r--', label='Predicted Prices (Next 30 Days)', linewidth=2)

# Format x-axis to show months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.title('30-Day Stock Price Forecast', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[59]:





# In[60]:


future_dates = pd.date_range(
    start=final_df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq='B'  # Business days only
)


# In[61]:


print(future_dates)  # Should show dates from 2024-08 to 2025-02


# In[62]:


# Add momentum and volatility features
final_df['Returns'] = final_df['Close'].pct_change()
final_df['Volatility'] = final_df['Returns'].rolling(14).std() * np.sqrt(14)  # 14-day vol
final_df['Momentum_5'] = final_df['Close'].pct_change(5)  # 5-day momentum


# In[63]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 7)),  # More units
    Dropout(0.2),  # Reduce overfitting
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')


# In[64]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
final_df_scaled = pd.DataFrame(scaler.fit_transform(final_df), columns=final_df.columns)


# In[65]:


# Add 10% of historical volatility to predictions
historical_vol = final_df['Returns'].std()
predicted_future_real = predicted_future_real * (1 + np.random.normal(0, 0.1 * historical_vol, len(predicted_future_real)))


# In[66]:


# After generating predictions, add volatility
volatility_factor = final_df['Returns'].std() * 0.5  # Adjust multiplier
predicted_future_real = [
    pred * (1 + np.random.normal(0, volatility_factor))
    for pred in predicted_future_real
]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(final_df['Close'].iloc[-90:], label='Historical Prices')
plt.plot(future_dates, predicted_future_real, 'r--', label='Predicted Prices (Adjusted)')
plt.title('30-Day Stock Price Forecast with Volatility')
plt.legend()
plt.grid()
plt.show()


# In[67]:


print("future_dates shape:", future_dates.shape)

print("predicted_future_real shape:", np.array(predicted_future_real).shape) # Convert to NumPy array


# In[68]:


# Trim longer array to match the shorter one
min_length = min(len(future_dates), len(predicted_future_real))
future_dates = future_dates[:min_length]
predicted_future_real = predicted_future_real[:min_length]


# In[69]:


predicted_future_real = np.array(predicted_future_real).flatten()  # Force 1D


# In[70]:


future_dates = future_dates.to_numpy()  # Convert to NumPy array


# In[71]:


import numpy as np

# Ensure 1D arrays and equal lengths
predicted_future_real = np.array(predicted_future_real).flatten()  # Shape: (30,)
future_dates = future_dates[:len(predicted_future_real)]  # Trim dates if needed

# Reshape predicted_future_real to match future_dates
predicted_future_real = predicted_future_real[:len(future_dates)]

# Plot
plt.figure(figsize=(14, 7))
plt.plot(
    future_dates,
    predicted_future_real,
    label='Predicted Prices + Volatility (Next 30 Days)',
    color='#FF6D00',
    linestyle='--',
    linewidth=2
)
plt.title('60-Day Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()


# In[72]:


# Save to CSV
future_df.to_csv("30_day_forecast.csv")
print("✅ Forecast saved to '30_day_forecast.csv'")


# In[73]:


# Create a combined DataFrame for dashboard plotting
full_plot_df = pd.concat([
    final_df[['Close']].iloc[-100:],  # Last 100 actual prices
    future_df[['Predicted_Close']]  # Only select 'Predicted_Close'
], axis=0)

# Optional: reset index for dashboard plotting
full_plot_df = full_plot_df.reset_index()


# In[74]:


model.save("lstm_stock_model.h5")
print("✅ Model saved to 'lstm_stock_model.h5'")


# In[75]:


pip install streamlit plotly


# In[76]:


import pandas as pd
import plotly.graph_objects as go

# Assuming 'final_df' contains the historical stock data
combined_df = pd.concat([final_df, future_df], axis=0)


# In[77]:


import streamlit as st
import plotly.graph_objects as go

# Title
st.title("📈 Stock Price Prediction Dashboard")

# Sidebar
stock = st.sidebar.selectbox("Choose Stock", ["AAPL"])  # Extend if you support more
days = st.sidebar.slider("Prediction Horizon (days)", 30, 90, 60)

# Load actual and predicted data
st.subheader(f"📊 Historical vs Predicted Prices for {stock}")

fig = go.Figure()
# 🐛 `actual_data` was not defined, assuming it should be `data` or `final_df`
# ✅ If using raw input data, use `data`
# ✅ If using feature-engineered data, use `final_df`
fig.add_trace(go.Scatter(x=final_df.index, y=final_df['Close'], name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], name='Predicted', line=dict(color='orange')))
st.plotly_chart(fig, use_container_width=True)

# Optionally display model summary or metrics
st.subheader("📄 Model Details")
st.text(model.summary())


# In[78]:


get_ipython().system('pip install streamlit pyngrok --quiet')


# In[79]:


get_ipython().run_cell_magic('writefile', 'dashboard.py', 'import streamlit as st\n\nst.title("📈 Stock Prediction Dashboard")\nst.write("This is your LSTM-based prediction dashboard.")\n# Add your charts, predictions, etc.\n')


# In[80]:


get_ipython().system('pip install streamlit')


# In[81]:


import os
os.listdir()


# In[82]:


get_ipython().system('cp lstm_predictions.csv final_predictions.csv')


# In[83]:


import pandas as pd

try:
    df = pd.read_csv('final_predictions.csv', index_col=0, parse_dates=True)
    print("✅ File loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    display(df.head())
except Exception as e:
    print("❌ Error loading file:", e)


# In[84]:


import pandas as pd

df = pd.read_csv("final_predictions.csv")
df.columns = ['Actual', 'Predicted']
df.index = pd.date_range(start='2023-01-01', periods=len(df))  # 🔁 Adjust start date
df.to_csv("final_predictions.csv")  # ✅ Save corrected version


# In[85]:


choice = "Actual vs Predicted"

if choice == "Actual vs Predicted":
    st.subheader("Actual vs Predicted Stock Prices")

    if not df.empty and 'Actual' in df.columns and 'Predicted' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted', line=dict(color='orange')))
        fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Required columns 'Actual' and 'Predicted' not found in data.")


# In[86]:


get_ipython().system('ls')


# In[87]:


import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from io import BytesIO
import base64

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="LSTM Stock Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    # 🐛 `__file__` is not defined in Jupyter/Colab
    # ✅ Use a relative path or absolute path instead
    # file_path = os.path.join(os.path.dirname(__file__), "final_predictions.csv")
    file_path = "final_predictions.csv"  # ✅ Assume file is in same directory
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("⚠️ 'final_predictions.csv' not found. Please generate or upload it.")
        return pd.DataFrame()
    return df

df = load_data()

# ... (rest of the code remains the same)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Model Overview", "Actual vs Predicted", "Forecast", "Residual Analysis", "Download Data"])

# ---------- Model Overview ----------
if choice == "Model Overview":
    st.title("📈 LSTM Stock Price Prediction Dashboard")
    st.markdown("""
    This dashboard presents the performance of a trained LSTM model that predicts stock prices.

    **Sections**:
    - Model Overview
    - Actual vs Predicted Prices
    - Forecast (30/60 Days)
    - Residual Analysis
    - Download Options
    """)

# ---------- Actual vs Predicted ----------
elif choice == "Actual vs Predicted":
    st.subheader("Actual vs Predicted Stock Prices")

    if not df.empty and 'Actual' in df.columns and 'Predicted' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted', line=dict(color='orange')))
        fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Required columns 'Actual' and 'Predicted' not found in data.")

# ---------- Forecast ----------
elif choice == "Forecast":
    st.subheader("📅 Forecast")
    st.markdown("Coming soon: display 30 and 60-day ahead LSTM forecasts here.")

# ---------- Residual Analysis ----------
elif choice == "Residual Analysis":
    st.subheader("📉 Residual Analysis")
    if 'Actual' in df.columns and 'Predicted' in df.columns:
        df['Residuals'] = df['Actual'] - df['Predicted']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Residuals'], name='Residuals', line=dict(color='green')))
        fig.update_layout(title="Residual Plot", xaxis_title="Date", yaxis_title="Residual")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Actual' and 'Predicted' columns not available to compute residuals.")

# ---------- Download Section ----------
elif choice == "Download Data":
    st.subheader("📥 Download Predictions")
    if not df.empty:
        towrite = BytesIO()
        df.to_csv(towrite)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        linko = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
        st.markdown(linko, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data available to download.")


# In[89]:


dashboard_code = """
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from io import BytesIO
import base64

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="LSTM Stock Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    # 🐛 `__file__` is not defined in Jupyter/Colab
    # ✅ Use a relative path or absolute path instead
    # file_path = os.path.join(os.path.dirname(__file__), "final_predictions.csv")
    file_path = "final_predictions.csv"  # ✅ Assume file is in same directory
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("⚠️ 'final_predictions.csv' not found. Please generate or upload it.")
        return pd.DataFrame()
    return df

df = load_data()

# ... (rest of the code remains the same)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Model Overview", "Actual vs Predicted", "Forecast", "Residual Analysis", "Download Data"])

# ---------- Model Overview ----------
if choice == "Model Overview":
    st.title("📈 LSTM Stock Price Prediction Dashboard")
    st.markdown(\"""
    This dashboard presents the performance of a trained LSTM model that predicts stock prices.

    **Sections**:
    - Model Overview
    - Actual vs Predicted Prices
    - Forecast (30/60 Days)
    - Residual Analysis
    - Download Options
    \""")

# ---------- Actual vs Predicted ----------
elif choice == "Actual vs Predicted":
    st.subheader("Actual vs Predicted Stock Prices")
    if not df.empty and 'Actual' in df.columns and 'Predicted' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted', line=dict(color='orange')))
        fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Required columns 'Actual' and 'Predicted' not found in data.")

# ---------- Forecast ----------
elif choice == "Forecast":
    st.subheader("📅 Forecast")
    st.markdown("Coming soon: display 30 and 60-day ahead LSTM forecasts here.")

# ---------- Residual Analysis ----------
elif choice == "Residual Analysis":
    st.subheader("📉 Residual Analysis")
    if 'Actual' in df.columns and 'Predicted' in df.columns:
        df['Residuals'] = df['Actual'] - df['Predicted']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Residuals'], name='Residuals', line=dict(color='green')))
        fig.update_layout(title="Residual Plot", xaxis_title="Date", yaxis_title="Residual")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Actual' and 'Predicted' columns not available to compute residuals.")

# ---------- Download Section ----------
elif choice == "Download Data":
    st.subheader("📥 Download Predictions")
    if not df.empty:
        towrite = BytesIO()
        df.to_csv(towrite)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        linko = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
        st.markdown(linko, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data available to download.")
"""

with open("dashboard.py", "w") as f:
    f.write(dashboard_code)


# In[90]:


from google.colab import files
files.upload()


# In[91]:


get_ipython().system('ls')


# In[92]:


# Import yfinance
import yfinance as yf

# Download stock data (replace 'AAPL' with your desired ticker)
stock_data = yf.download("AAPL", start="2015-01-01", end="2024-12-31")

# Now you can save it to CSV
stock_data.to_csv('final_predictions.csv')
print("✅ final_predictions.csv saved.")


# In[93]:


get_ipython().system('ls')


# 

# In[ ]:





# In[96]:


get_ipython().system('pkill -f ngrok')


# In[99]:


import os

@st.cache_data
def load_data():
    try:
        # Use absolute path to ensure it finds the file
        file_path = os.path.join(os.getcwd(), "final_predictions.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("⚠️ 'final_predictions.csv' not found at: " + file_path)
        return pd.DataFrame()
    return df


# In[100]:


get_ipython().system('ls -l')


# In[101]:


get_ipython().system('ls -l final_predictions.csv')


# In[102]:


get_ipython().system('pip install streamlit pyngrok --quiet')


# In[104]:


import pandas as pd

df = pd.read_csv("final_predictions.csv")
print(df.columns)
df.head()


# In[112]:


import pandas as pd

df_raw = pd.read_csv("final_predictions.csv", header=None)
df_raw.head(10)


# In[113]:


df = pd.read_csv("final_predictions.csv", skiprows=2, usecols=[0, 1, 2])
df.columns = ['Date', 'Actual', 'Predicted']


# In[114]:


print(df.columns)
print(df.head())


# In[115]:


df_raw = pd.read_csv("final_predictions.csv", header=None)
for i in range(5):
    print(f"\nRow {i}:\n", df_raw.iloc[i])


# In[116]:


import pandas as pd

# Skip first 3 rows (rows 0, 1, 2)
df = pd.read_csv("final_predictions.csv", skiprows=3, usecols=[0, 1, 2])

# Rename columns properly
df.columns = ['Date', 'Actual', 'Predicted']

# Confirm
print(df.head())


# In[117]:


import pandas as pd

# Skip first 3 metadata rows
df = pd.read_csv("final_predictions.csv", skiprows=3, usecols=[0, 1, 2])  # Adjust columns if needed

# Rename columns clearly
df.columns = ['Date', 'Actual', 'Predicted']

# Convert 'Date' to datetime if necessary
df['Date'] = pd.to_datetime(df['Date'])

# Optional: set Date as index (not mandatory unless your code needs it)
# df.set_index('Date', inplace=True)

# Save back cleaned file
df.to_csv("final_predictions.csv", index=False)

print("✅ final_predictions.csv cleaned and saved.")
print(df.head())


# In[118]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Actual'], label='Actual', color='blue')
plt.plot(df.index, df['Predicted'], label='Predicted', color='orange')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("📈 Predicted vs Actual Stock Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[120]:


import pandas as pd

# Skip the extra header and metadata rows
df = pd.read_csv("final_predictions.csv", skiprows=3, usecols=[0, 1, 2])  # Adjust if needed

# Rename columns to match dashboard expectations
df.columns = ['Date', 'Actual', 'Predicted']

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where Date or values are missing (just in case)
df.dropna(subset=['Date', 'Actual', 'Predicted'], inplace=True)

# Save back the cleaned file (overwrite)
df.to_csv("final_predictions.csv", index=False)

print("✅ Dashboard-ready final_predictions.csv saved!")
print(df.head())


# In[122]:


df.head()


# In[119]:


get_ipython().system('cp /mnt/data/final_predictions.csv final_predictions.csv')


# In[123]:


from google.colab import files
uploaded = files.upload()


# In[124]:


get_ipython().system('ls -l')


# In[125]:


import pandas as pd

df = pd.read_csv("final_predictions.csv")
print("🔍 Column names:", df.columns.tolist())
df.head()


# In[126]:


# 🛠 Replace this with your actual predictions DataFrame
pred_df = pd.DataFrame({
    "Date": pd.date_range(start="2025-04-01", periods=5, freq="D"),
    "Actual": [184.52, 183.97, 185.32, 186.14, 185.78],
    "Predicted": [185.12, 184.65, 185.44, 186.02, 185.90]
})

# ✅ Save it correctly
pred_df.to_csv("final_predictions.csv", index=False)


# In[127]:


df.head()


# In[130]:


yfrom pyngrok import ngrok, conf

# Kill all active tunnels
ngrok.kill()

# Check if the process is killed
try:
    ngrok_process = ngrok.get_ngrok_process()
    if ngrok_process.proc.poll() is None:  # Process is still running
        st.error("⚠️ ngrok process is still running. Please try again.")
    else:
        # Restart the dashboard tunnel
        public_url = ngrok.connect(8501)
        print("🚀 Dashboard is live at:", public_url)

except Exception as e:
    st.error(f"⚠️ Error connecting to ngrok: {e}")


# In[ ]:


pred_df.to_csv("final_predictions.csv", index=False)


# In[ ]:


get_ipython().system('ls -l final_predictions.csv')


# In[131]:


get_ipython().system('streamlit run dashboard.py')


# In[132]:


get_ipython().system('streamlit run dashboard.py &> /dev/null &')

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🚀 Your dashboard is live at:", public_url)


# In[133]:


authtoken: YOUR_AUTH_TOKEN
tunnels:
  streamlit:
    addr: 8501
    proto: http


# In[ ]:





# In[134]:


get_ipython().system('ngrok config add-authtoken 2vuSOrGwa2rfMtq76gvpn8PNOp1_2iyR5v3KahjVLvua9spaw')


# In[137]:


from pyngrok import ngrok

# Optional: Set new authtoken
ngrok.set_auth_token("2vuSOrGwa2rfMtq76gvpn8PNOp1_2iyR5v3KahjVLvua9spaw")

# Kill any previous session just to be safe
ngrok.kill()

# Start tunnel
public_url = ngrok.connect(8501)
print("🚀 Your dashboard is live at:", public_url)

