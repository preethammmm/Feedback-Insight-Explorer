import pandas as pd
from textblob import TextBlob

df = pd.read_csv('clustered_reviews.csv')

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0

df['sentiment'] = df['reviews.text'].apply(get_sentiment)

df.to_csv('clustered_reviews_with_sentiment.csv', index=False)
print("Sentiment analysis complete. Results saved to clustered_reviews_with_sentiment.csv")
