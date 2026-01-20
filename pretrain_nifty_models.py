import joblib
from modules.stock_data import get_stock_data
from modules.features import create_features
from modules.model import train_model

# NIFTY 50 tickers you provided
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

days = 365  # Use past 1 year data

for company, ticker in nifty50_tickers.items():
    print(f"Training model for {company} ({ticker})...")
    
    # Fetch stock data
    df = get_stock_data(ticker).tail(days)
    if df.empty:
        print(f"⚠️ Stock data not available for {company}. Skipping...")
        continue
    
    # Create features
    df_features = create_features(df)
    
    # Train model
    model, X_test, y_test = train_model(df_features)
    
    # Save the trained model
    filename = f"models/{ticker.replace('.','_')}_model.pkl"
    joblib.dump(model, filename)
    print(f"✅ Saved model to {filename}")
