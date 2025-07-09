import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# === Load model and scalers ===
model = load_model("model/model.h5")
feature_scaler = joblib.load("model/feature_scaler.save")
target_scaler = joblib.load("model/target_scaler.save")

# === Fetch and prepare stock data ===
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.rename(columns={
        "Open": "Open_yfin", "High": "High_yfin", "Low": "Low_yfin",
        "Close": "Close_yfin", "Volume": "Volume_yfin"
    }, inplace=True)

    df['SMA_10'] = df['Close_yfin'].rolling(window=10).mean()
    df['SMA_20'] = df['Close_yfin'].rolling(window=20).mean()
    df['EMA_10'] = df['Close_yfin'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close_yfin'].ewm(span=20, adjust=False).mean()
    df['Rolling_STD_10'] = df['Close_yfin'].rolling(window=10).std()
    df['Rolling_Max_10'] = df['Close_yfin'].rolling(window=10).max()
    df['Rolling_Min_10'] = df['Close_yfin'].rolling(window=10).min()
    df['Momentum_10'] = df['Close_yfin'] - df['Close_yfin'].shift(10)

    delta = df['Close_yfin'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close_yfin'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close_yfin'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma = df['Close_yfin'].rolling(window=20).mean()
    std = df['Close_yfin'].rolling(window=20).std()
    df['Bollinger_Width'] = (2 * std) / sma

    df.dropna(inplace=True)
    return df

# === Preprocess input ===
def preprocess(df):
    features = [
        'Close_yfin', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
        'Rolling_STD_10', 'Rolling_Max_10', 'Rolling_Min_10',
        'Momentum_10', 'RSI_14', 'MACD', 'Signal_Line', 'Bollinger_Width'
    ]
    last_60 = df[features].tail(60).values
    X_scaled = feature_scaler.transform(last_60)
    last_close = float(df['Close_yfin'].iloc[-1])
    return np.expand_dims(X_scaled, axis=0), last_close

# === Streamlit UI ===
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Predictor & Investment Simulator")

ticker = st.text_input("Enter Stock Symbol:", "AAPL")
investment = st.number_input("💵 Investment Amount ($):", value=1000.0)

df = None
if st.button("Predict Next Day Price"):
    try:
        df = fetch_data(ticker)
        X_input, last_price = preprocess(df)
        pred_scaled = model.predict(X_input)[0][0]

        dummy = np.zeros((1, len(feature_scaler.feature_names_in_)))
        dummy[0][0] = pred_scaled
        predicted_price = float(target_scaler.inverse_transform(dummy)[0][0])

        st.success("Prediction Complete!")
        st.metric("📉 Last Close Price", f"${last_price:.2f}")
        st.metric("📈 Predicted Next Price", f"${predicted_price:.2f}")

        # === Buy / Hold / Sell Signal ===
        st.subheader("🧠 Strategy Recommendation")
        if predicted_price > last_price * 1.01:
            st.success("📈 Suggested Action: **BUY**")
        elif predicted_price < last_price * 0.99:
            st.error("📉 Suggested Action: **SELL**")
        else:
            st.info("⏸ Suggested Action: **HOLD**")

        # === Investment simulation ===
        profit = (predicted_price - last_price) * (investment / last_price)
        st.subheader("💰 Investment Simulation")
        st.write(f"If you invest **${investment:.2f}** now:")
        st.write(f"**Predicted Value Tomorrow:** ${investment + profit:.2f}")
        if profit > 0:
            st.success(f"📈 Estimated Profit: ${profit:.2f}")
        else:
            st.error(f"📉 Estimated Loss: ${-profit:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# === Plot Historical Chart ===
if df is not None:
    st.subheader(f"📊 Historical Close Price for {ticker}")
    try:
        st.line_chart(df['Close_yfin'].tail(60))
    except Exception as e:
        st.warning(f"Could not plot chart: {e}")



   






