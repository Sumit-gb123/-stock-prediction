# from newsapi import NewsApiClient
# from textblob import TextBlob
# import pandas as pd

# NEWS_API_KEY = "2f10871cc358456f858ae4f32f5dfc29"

# def get_news_sentiment(stock_name='NIFTY 50', max_articles=20):
#     newsapi = NewsApiClient(api_key=NEWS_API_KEY)
#     all_articles = newsapi.get_everything(
#         q=stock_name,
#         language='en',
#         sort_by='publishedAt',
#         page_size=max_articles
#     )

#     sentiments = []
#     for article in all_articles.get('articles', []):  # safe access
#         title = article.get('title') or ''
#         description = article.get('description') or ''
#         text = title + " " + description
#         polarity = TextBlob(text).sentiment.polarity
#         date = article.get('publishedAt', '')[:10]  # YYYY-MM-DD
#         if date:  # only add if date exists
#             sentiments.append({'date': date, 'sentiment': polarity})

#     if len(sentiments) == 0:
#         # No articles found, return empty DataFrame with correct columns
#         return pd.DataFrame(columns=['date', 'sentiment'])

#     df_sentiment = pd.DataFrame(sentiments)
#     df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
#     df_sentiment = df_sentiment.groupby('date', as_index=False).mean()
    
#     return df_sentiment

from newsapi import NewsApiClient
from textblob import TextBlob

NEWS_API_KEY = "2f10871cc358456f858ae4f32f5dfc29"

def get_news_sentiment(stock_name, max_articles=30):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    news = newsapi.get_everything(
        q=stock_name,
        language="en",
        sort_by="relevancy",
        page_size=max_articles
    )

    sentiments = []
    for article in news["articles"]:
        text = f"{article['title']} {article.get('description','')}"
        polarity = TextBlob(text).sentiment.polarity
        sentiments.append(polarity)

    if len(sentiments) == 0:
        return 0.0

    return round(sum(sentiments) / len(sentiments), 3)
