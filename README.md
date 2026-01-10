Advanced NIFTY 50 Stock Predictor

Project Overview::
This project is an advanced stock market prediction dashboard for NIFTY 50 companies. It combines historical stock data with real-time news sentiment analysis to provide predictions on whether a stock price is likely to go up, down, or hold. The project uses machine learning (XGBoost) to analyze stock trends, along with moving averages and returns as technical features.

Key Features:

Real-time fetching of historical stock prices using Yahoo Finance (yfinance).

Sentiment analysis from news articles using NewsAPI and TextBlob.

Technical feature calculation: Moving Averages (MA5, MA10), daily returns, etc.

Machine Learning model (XGBoost) for stock price movement prediction.

Interactive Streamlit dashboard with:

Stock candlestick charts

Moving averages visualization

Latest prediction and recommendation (Buy, Sell, Hold)

Sentiment score and market sentiment

Model confidence display

Recent stock trading data

Technologies & Libraries Used:

Python 3.13

Streamlit (Web Dashboard)

Plotly (Interactive Charts)

Pandas & NumPy (Data manipulation)

Scikit-learn (Data preprocessing & model evaluation)

XGBoost (Prediction model)

yfinance (Stock data)

NewsAPI & TextBlob (News sentiment analysis)

System Architecture:

Data Collection: Fetch historical stock prices and news sentiment.

Feature Engineering: Compute technical indicators and daily returns.

Model Training: Train XGBoost classifier using features and target (next-day price direction).

Prediction: Use latest data to predict price movement and combine with sentiment for recommendation.

Visualization: Display candlestick charts, moving averages, sentiment, and prediction in Streamlit.

