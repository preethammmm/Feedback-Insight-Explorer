# import streamlit as st
# import pandas as pd
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import spacy
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from textblob import TextBlob
# import gensim
# from gensim import corpora
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # --- Page Config (MUST BE FIRST) ---
# st.set_page_config(page_title="Customer Feedback Cluster Explorer", layout="wide")
# st.title("Customer Feedback Cluster Explorer")

# # --- User-Defined Aspects ---
# with st.sidebar:
#     st.header("Settings")
#     custom_aspects = st.text_input(
#         "Comma-separated aspects (e.g., delivery, pricing, service):",
#         "delivery, pricing, service, quality, website"
#     )
#     aspects = [a.strip() for a in custom_aspects.split(",")]
#     similarity_threshold = st.slider("Aspect Similarity Threshold", 0.5, 0.9, 0.7)

# # --- Load spaCy model with word vectors ---
# try:
#     nlp = spacy.load("en_core_web_lg")
# except OSError:
#     st.error("Please download the spaCy model: `python -m spacy download en_core_web_lg`")
#     st.stop()

# def sentiment_label(score):
#     if score > 0.1:
#         return "Positive", "green", "😊"
#     elif score < -0.1:
#         return "Negative", "red", "😞"
#     else:
#         return "Neutral", "gray", "😐"

# def extract_aspect(text):
#     """Detect aspect using semantic similarity to predefined aspects."""
#     doc = nlp(text)
#     max_similarity = 0
#     best_aspect = "Other"
    
#     for token in doc:
#         for aspect in aspects:
#             aspect_doc = nlp(aspect)
#             similarity = token.similarity(aspect_doc)
#             if similarity > max_similarity and similarity > similarity_threshold:
#                 max_similarity = similarity
#                 best_aspect = aspect.capitalize()
    
#     return best_aspect

# def get_lda_topics(texts, num_topics=3, num_words=5):
#     """Generate LDA topics with improved parameters."""
#     tokenized = [text.split() for text in texts]
    
#     # Handle empty tokenized texts
#     if not tokenized or all(len(t) == 0 for t in tokenized):
#         return ["No topics found"]
    
#     dictionary = corpora.Dictionary(tokenized)
#     corpus = [dictionary.doc2bow(text) for text in tokenized]
    
#     # Handle empty corpus
#     if not corpus:
#         return ["No topics found"]
    
#     lda_model = gensim.models.LdaModel(
#         corpus=corpus,
#         id2word=dictionary,
#         num_topics=num_topics,
#         random_state=42,
#         passes=15,
#         alpha=0.1,
#         eta='auto'
#     )
    
#     return [topic[1] for topic in lda_model.print_topics(num_words=num_words)]

# if uploaded_file := st.file_uploader("Upload a CSV file", type=["csv"]):
#     df = pd.read_csv(uploaded_file)
#     TEXT_COLUMN = 'reviews.text' if 'reviews.text' in df.columns else df.columns[0]

#     # --- Preprocessing ---
#     def clean_text(text):
#         text = str(text)
#         text = re.sub(r'http\S+', '', text)
#         text = re.sub(r'[^\w\s]', '', text)
#         return text.strip()
    
#     df['cleaned'] = df[TEXT_COLUMN].astype(str).apply(clean_text)
#     df['preprocessed'] = df['cleaned'].apply(lambda x: " ".join([token.lemma_.lower() for token in nlp(x) if not token.is_stop and token.is_alpha]))

#     # --- TF-IDF & Clustering ---
#     tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
#     X = tfidf.fit_transform(df['preprocessed'])
#     k = min(5, len(df))
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     df['cluster'] = kmeans.fit_predict(X)

#     # --- Sentiment & Aspect Analysis ---
#     df['sentiment'] = df[TEXT_COLUMN].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
#     df['aspect'] = df['preprocessed'].apply(extract_aspect)

#     # --- LDA Topics ---
#     cluster_topics = {}
#     for cluster in df['cluster'].unique():
#         cluster_texts = df[df['cluster'] == cluster]['preprocessed'].tolist()
#         topics = get_lda_topics(cluster_texts) if len(cluster_texts) >= 10 else ["Insufficient data"]
#         cluster_topics[cluster] = topics

#     # --- UI ---
#     clusters = sorted(df['cluster'].unique())
#     selected_cluster = st.sidebar.selectbox("Select cluster", clusters)
#     cluster_df = df[df['cluster'] == selected_cluster]

#     st.write(f"## Cluster {selected_cluster} ({len(cluster_df)} reviews)")
    
#     # Aspect Distribution
#     st.write("### Aspect Distribution")
#     aspect_counts = cluster_df['aspect'].value_counts()
#     st.bar_chart(aspect_counts)

#     # LDA Topics
#     st.write("### Key Topics")
#     for topic in cluster_topics.get(selected_cluster, ["No topics available"]):
#         st.markdown(f"- {topic}")

#     # Word Cloud
#     st.write("### Topic Word Cloud")
#     if cluster_text := " ".join(cluster_df['preprocessed'].astype(str)).strip():
#         st.image(WordCloud(width=800, height=400).generate(cluster_text).to_array())
#     else:
#         st.info("No text available for wordcloud")

#     # Sentiment & Reviews
#     avg_sentiment = cluster_df['sentiment'].mean()
#     label, color, emoji = sentiment_label(avg_sentiment)
#     st.write(f"### Average Sentiment: {emoji} **<span style='color:{color}'>{label} ({avg_sentiment:.2f})</span>**", unsafe_allow_html=True)
    
#     st.write("### Sample Reviews by Aspect")
#     for aspect in aspect_counts.index:
#         with st.expander(f"{aspect} ({aspect_counts[aspect]})"):
#             for _, row in cluster_df[cluster_df['aspect'] == aspect].head(3).iterrows():
#                 sent_label, sent_color, sent_emoji = sentiment_label(row['sentiment'])
#                 st.markdown(f"{sent_emoji} <span style='color:{sent_color}'>{row[TEXT_COLUMN]}</span>", unsafe_allow_html=True)