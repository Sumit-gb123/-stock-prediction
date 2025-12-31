# import yfinance as yf
# import pandas as pd

# def get_stock_data(ticker='NIFTY50.NS', start='2020-01-01', end=None):
#     if end is None:
#         end = pd.Timestamp.today().strftime('%Y-%m-%d')
#     data = yf.download(ticker, start=start, end=end)
#     data.reset_index(inplace=True)
#     return data
import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False)

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)

    # If MultiIndex columns exist → flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

