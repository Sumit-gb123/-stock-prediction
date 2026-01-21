import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

from modules.stock_data import get_stock_data
from modules.sentiment import get_news_sentiment
from modules.features import create_features
from modules.model import train_model

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
.metric-card { background: #111827; border-radius: 14px; padding: 22px; border: 1px solid #374151;
              box-shadow: 0 8px 24px rgba(0,0,0,0.35); text-align: center; }
.section-divider { margin: 40px 0; border-top: 1px solid #374151; }
section[data-testid="stSidebar"] { background: #020617; }
</style>
""", unsafe_allow_html=True)

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="MarketPulse AI", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>📈 MarketPulse AI</h1>
<p style='text-align:center; color:#9CA3AF; font-size:18px;'>
Intelligent Market Forecasting, Risk & Sentiment Analytics Platform
</p>
""", unsafe_allow_html=True)

# ===================== NIFTY 50 =====================
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
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Sun Pharma": "SUNPHARMA.NS"
}

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Configuration")
company_name = st.sidebar.selectbox("Select NIFTY 50 Company", list(nifty50_tickers.keys()))
ticker = nifty50_tickers[company_name]
days = st.sidebar.slider("Past Days", 100, 1500, 365)

# ===================== LOADING ANIMATION =====================
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.info("Loading dashboard and calculating predictions...")
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.01)  # small animation

# ===================== FETCH DATA =====================
stock_df = get_stock_data(ticker).tail(days)
if stock_df.empty:
    st.error(f"Stock data for {company_name} not available.")
    st.stop()

stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df['Return'] = stock_df['Close'].pct_change()

# ===================== SENTIMENT =====================
try:
    sentiment_score, negative_news = get_news_sentiment(company_name)
except Exception as e:
    st.warning(f"News sentiment unavailable: {e}")
    sentiment_score = 0.0
    negative_news = []

# ===================== FEATURES =====================
df = create_features(stock_df)
if len(df) < 30:
    st.error("Not enough data.")
    st.stop()

# ===================== TRAIN MODEL =====================
model, X_test, y_test = train_model(df)
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)

# ===================== PREDICTION =====================
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return']
latest_features = df[features].iloc[-1].values.reshape(1, -1)
pred = model.predict(latest_features)[0]
prob = model.predict_proba(latest_features)[0]
confidence = float(np.max(prob) * 100)

# ===================== RISK / VOLATILITY =====================
volatility = stock_df['Return'].rolling(20).std().iloc[-1] * 100
risk = "🟢 Low Risk" if volatility < 1 else "🟡 Medium Risk" if volatility < 2 else "🔴 High Risk"

# ===================== TREND STRENGTH =====================
ma_diff = abs(df['MA5'].iloc[-1] - df['MA10'].iloc[-1])
trend = "Weak" if ma_diff < 0.5 else "Moderate" if ma_diff < 1.5 else "Strong"

# ===================== MARKET REGIME =====================
short_ma = stock_df['Close'].rolling(10).mean().iloc[-1]
long_ma = stock_df['Close'].rolling(50).mean().iloc[-1]
regime = "🐂 Bull Market" if short_ma > long_ma and volatility < 2 else "🐻 Bear Market" if short_ma < long_ma and volatility > 2 else "⚖️ Sideways Market"

# ===================== FINAL DECISION =====================
if confidence < 55:
    recommendation = "⚠️ Low Confidence – Avoid Trading"
elif pred == 1 and sentiment_score > 0:
    recommendation = "📈 BUY"
elif pred == 0 and sentiment_score < 0:
    recommendation = "📉 SELL"
else:
    recommendation = "⏸ HOLD"

# ===================== CLEAR LOADING PLACEHOLDER =====================
loading_placeholder.empty()  # removes the animation once calculations are done

# ===================== METRIC CARDS =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"<div class='metric-card'><h3>Stock</h3><h2>{company_name}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><h3>Last Close</h3><h2>₹ {round(stock_df['Close'].iloc[-1],2)}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><h3>Risk Level</h3><h2>{risk}</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><h3>Trend</h3><h2>{trend}</h2></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='metric-card'><h3>Market Regime</h3><h2>{regime}</h2></div>", unsafe_allow_html=True)

# ===================== SENTIMENT DASHBOARD =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📰 News Sentiment Intelligence")
s1, s2, s3 = st.columns(3)

with s1:
    st.metric("Sentiment Score", round(sentiment_score, 3))
with s2:
    if sentiment_score > 0.15:
        st.success("Strong Positive 🟢")
    elif sentiment_score > 0:
        st.info("Mild Positive 🔵")
    elif sentiment_score < -0.15:
        st.error("Strong Negative 🔴")
    else:
        st.warning("Neutral ⚪")
with s3:
    st.metric("Market Bias", "Bullish Bias" if sentiment_score > 0 else "Bearish Bias")

if negative_news:
    with st.expander("🔻 Negative News Impacting Stock"):
        for news in negative_news[:5]:
            st.write("•", news)

# ===================== REST OF YOUR DASHBOARD =====================
# (You can include charts, Monte Carlo simulation, final signals, etc., just like in your previous full code)


# ===================== TOMORROW PRICE ESTIMATION =====================
last_close = stock_df['Close'].iloc[-1]
short_ma = stock_df['Close'].rolling(5).mean().iloc[-1]
long_ma = stock_df['Close'].rolling(20).mean().iloc[-1]
trend_factor = 0.005 if short_ma > long_ma else -0.005 if short_ma < long_ma else 0.0
vol_factor = stock_df['Return'].rolling(20).std().iloc[-1]
sent_factor = 0.003 * np.sign(sentiment_score)
predicted_return = trend_factor + np.random.normal(0, vol_factor) + sent_factor
predicted_price = last_close * (1 + predicted_return)
support = stock_df['Low'].rolling(20).min().iloc[-1]
resistance = stock_df['High'].rolling(20).max().iloc[-1]
predicted_price = max(support, min(predicted_price, resistance * 1.02))

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🔮 Tomorrow Price Estimation")
st.metric("Estimated Price", f"₹ {predicted_price:.2f}")

st.markdown(f"""
**Prediction Logic Explained:**

- **Trend Factor:** {'Bullish' if trend_factor>0 else 'Bearish' if trend_factor<0 else 'Neutral'} ({trend_factor*100:.2f}% change)  
- **Volatility Factor:** Daily variation based on last 20 days returns ({vol_factor:.3f})  
- **Sentiment Factor:** {'Positive' if sent_factor>0 else 'Negative' if sent_factor<0 else 'Neutral'} ({sent_factor*100:.2f}% change)  
- **Support & Resistance:** Price limited between ₹{support:.2f} and ₹{resistance*1.02:.2f}
""")

# ===================== FINAL SIGNAL =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-card">
    <h2>📌 Final Trading Signal</h2>
    <h1>{recommendation}</h1>
    <p style="color:#9CA3AF;">ML + Sentiment + Risk + Regime Detection</p>
</div>
""", unsafe_allow_html=True)

# ===================== MODEL PERFORMANCE =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🧪 Model Backtesting Performance")
st.metric("Directional Accuracy", f"{accuracy*100:.2f}%")
st.text(classification_report(y_test, y_pred_test))

# ===================== CONFIDENCE =====================
st.subheader("🎯 Prediction Confidence")
st.progress(int(confidence))
st.caption(f"Confidence: {confidence:.2f}%")

# ===================== MONTE CARLO SIMULATION =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📉 Monte Carlo Price Forecast (30 Days)")
mc_days = 30
mc_runs = 100
paths = []
for _ in range(mc_runs):
    prices = [last_close]
    for _ in range(mc_days):
        prices.append(prices[-1] * (1 + np.random.normal(0, stock_df['Return'].std())))
    paths.append(prices)
mc_df = pd.DataFrame(paths).T
st.line_chart(mc_df)

# ===================== SUPPORT & RESISTANCE =====================
support = stock_df['Low'].rolling(20).min().iloc[-1]
resistance = stock_df['High'].rolling(20).max().iloc[-1]

# ===================== CHART =====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=stock_df['Date'],
    open=stock_df['Open'],
    high=stock_df['High'],
    low=stock_df['Low'],
    close=stock_df['Close']
))
fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'].rolling(5).mean(), name="MA5"))
fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'].rolling(10).mean(), name="MA10"))
fig.add_hline(y=support, line_dash="dot", line_color="green",
              annotation_text="Support", annotation_position="bottom left")
fig.add_hline(y=resistance, line_dash="dot", line_color="red",
              annotation_text="Resistance", annotation_position="top left")
fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ===================== DATA =====================
st.subheader("📋 Recent Trading Data")
st.dataframe(stock_df.tail(10))

# ===================== FOOTER =====================
st.markdown("""
<p style="text-align:center; color:#9CA3AF; margin-top:40px;">
⚠️ Educational & Research Purpose Only<br>
MarketPulse AI does not provide financial advice
</p>
""", unsafe_allow_html=True)
