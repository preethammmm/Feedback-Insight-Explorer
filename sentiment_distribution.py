import pandas as pd
df = pd.read_csv('clustered_reviews_with_sentiment.csv')
print(df['sentiment'].describe())
print(df['sentiment'].hist())
print(df[df['sentiment'] < 0][['reviews.text', 'sentiment']].head(10))
