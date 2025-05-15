import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('7817_1.csv')

# If your CSV has a different text column, update this
TEXT_COLUMN = 'reviews.text' if 'reviews.text' in df.columns else df.columns[0]

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

df['cleaned'] = df[TEXT_COLUMN].astype(str).apply(clean_text)

# NLP preprocessing with spaCy
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

df['preprocessed'] = df['cleaned'].apply(preprocess)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['preprocessed'])

# KMeans clustering (choose k by silhouette or elbow method; here k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Save clustered data for Streamlit app
df.to_csv('clustered_reviews.csv', index=False)
print(f"Clustering complete. Results saved to clustered_reviews.csv")
