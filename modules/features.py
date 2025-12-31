# import pandas as pd
# import numpy as np

# def create_features(stock_df, sentiment_df):
#     df = stock_df.copy()

#     # ---------------- Flatten MultiIndex columns if any ----------------
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

#         rename_dict = {}
#         for col in df.columns:
#             if 'Date' in col:
#                 rename_dict[col] = 'Date'
#             if 'Close' in col:
#                 rename_dict[col] = 'Close'
#             if 'Open' in col:
#                 rename_dict[col] = 'Open'
#             if 'High' in col:
#                 rename_dict[col] = 'High'
#             if 'Low' in col:
#                 rename_dict[col] = 'Low'
#             if 'Volume' in col:
#                 rename_dict[col] = 'Volume'
#         df.rename(columns=rename_dict, inplace=True)

#     # ---------------- Reset index ----------------
#     df = df.reset_index() if isinstance(df.index, pd.MultiIndex) or 'Date' not in df.columns else df

#     # ---------------- Ensure Date is datetime ----------------
#     df['Date'] = pd.to_datetime(df['Date'])

#     # ---------------- Technical Features ----------------
#     df['MA5'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
#     df['MA10'] = df['Close'].rolling(window=10).mean().fillna(method='bfill')
#     df['Return'] = df['Close'].pct_change().fillna(0)

#     # ---------------- Merge Sentiment ----------------
#     if not sentiment_df.empty:
#         sentiment_df = sentiment_df.copy()
#         sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
#         df = df.merge(sentiment_df, left_on='Date', right_on='date', how='left')
#         df['sentiment'].fillna(0, inplace=True)
#     else:
#         df['sentiment'] = 0

#     # ---------------- Target variable ----------------
#     df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

#     # Drop rows if essential columns missing
#     df.dropna(subset=['Close', 'Target'], inplace=True)

#     return df
import pandas as pd

def create_features(stock_df):
    df = stock_df.copy()

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['Return'] = df['Close'].pct_change()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df
