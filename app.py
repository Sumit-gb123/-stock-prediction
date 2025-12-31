import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from modules.stock_data import get_stock_data
from modules.sentiment import get_news_sentiment
from modules.features import create_features
from modules.model import train_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Advanced NIFTY 50 Stock Predictor",
    layout="wide"
)

st.title("📈 Advanced NIFTY 50 Stock Predictor")

# ---------------- NIFTY 50 Companies ----------------
nifty50_tickers = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",

    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",

    "Bharti Airtel": "BHARTIARTL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",

    "Larsen & Toubro": "LT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",

    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",

    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",

    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy’s Labs": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",

    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS"
}

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configuration")

company_name = st.sidebar.selectbox(
    "Select NIFTY 50 Company",
    list(nifty50_tickers.keys())
)

ticker = nifty50_tickers[company_name]

days = st.sidebar.slider(
    "Past Days to Fetch",
    min_value=100,
    max_value=1500,
    value=365
)

# ---------------- Fetch Stock Data ----------------
st.info(f"Fetching stock data for {company_name}...")
stock_df = get_stock_data(ticker).tail(days)

if stock_df.empty:
    st.error("Stock data not available.")
    st.stop()

stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# 🔧 FIX: Add Return column to stock_df (needed for metrics)
stock_df['Return'] = stock_df['Close'].pct_change()

# ---------------- Fetch News Sentiment ----------------
st.info(f"Fetching news sentiment for {company_name}...")
sentiment_score = get_news_sentiment(company_name)

# ---------------- Create Features (ML Data) ----------------
df = create_features(stock_df)

if len(df) < 30:
    st.error("Not enough historical data to train model.")
    st.stop()

# ---------------- Train Model ----------------
st.info("Training prediction model...")
model, X_test, y_test = train_model(df)

# ---------------- Latest Prediction ----------------
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return']

latest_features = df[feature_cols].iloc[-1].values.reshape(1, -1)

pred = model.predict(latest_features)[0]
prob = model.predict_proba(latest_features)[0]
confidence = float(np.max(prob) * 100)

# ---------------- Final Decision Logic ----------------
if pred == 1 and sentiment_score > 0.05:
    recommendation = "📈 BUY"
elif pred == 0 and sentiment_score < -0.05:
    recommendation = "📉 SELL"
else:
    recommendation = "⏸ HOLD"

# ---------------- TOP METRIC CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Stock", company_name)
col2.metric("Last Close (₹)", round(stock_df['Close'].iloc[-1], 2))
col3.metric(
    "Daily Return (%)",
    round(stock_df['Return'].iloc[-1] * 100, 2)
    if not pd.isna(stock_df['Return'].iloc[-1]) else 0.0
)
col4.metric("Prediction", "UP ⬆️" if pred == 1 else "DOWN ⬇️")

# ---------------- Sentiment Analysis ----------------
st.subheader("📰 News Sentiment Analysis")
st.metric("Sentiment Score", round(sentiment_score, 3))

if sentiment_score > 0.1:
    st.success("Market Sentiment: Strongly Positive 🟢")
elif sentiment_score > 0:
    st.info("Market Sentiment: Slightly Positive 🔵")
elif sentiment_score < -0.1:
    st.error("Market Sentiment: Strongly Negative 🔴")
else:
    st.warning("Market Sentiment: Neutral ⚪")

# ---------------- Prediction Result ----------------
st.subheader("📊 Prediction Result")
st.success(f"Price Direction: {'UP ⬆️' if pred == 1 else 'DOWN ⬇️'}")
st.success(f"Final Recommendation: {recommendation}")

# ---------------- Model Confidence ----------------
st.subheader("🎯 Model Confidence")
st.progress(int(confidence))
st.caption(f"Confidence Level: {confidence:.2f}%")

# ---------------- Plot Stock Chart ----------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=stock_df['Date'],
    open=stock_df['Open'],
    high=stock_df['High'],
    low=stock_df['Low'],
    close=stock_df['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['Close'].rolling(5).mean(),
    mode='lines',
    name='MA 5'
))

fig.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['Close'].rolling(10).mean(),
    mode='lines',
    name='MA 10'
))

fig.update_layout(
    title=f"{company_name} Stock Price with Moving Averages",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Recent Data ----------------
st.subheader("📋 Recent Trading Data")
st.dataframe(stock_df.tail(10))

# ---------------- Footer ----------------
st.caption(
    "⚠️ Educational Project | Uses ML + News Sentiment | Not Financial Advice"
)
