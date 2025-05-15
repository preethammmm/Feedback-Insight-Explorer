import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import re
import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gensim
from gensim import corpora
import warnings
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch  # For PyTorch support
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuring the page of the streamlit
st.set_page_config(page_title="Advanced Feedback Analyzer", layout="wide")
st.title("📊 Advanced Customer Feedback Analysis")

# Configuring the sidebar(left-panel)
with st.sidebar:
    st.header("⚙️ Settings")
    # Choosing the sentiment analysis from the options
    st.subheader("Sentiment Analysis")
    sentiment_model = st.radio("Choose Model:", ["VADER", "BERT"])
    
    # Select the Topic Labeling by GPT4
    st.subheader("Topic Labeling")
    use_llm_labeling = st.checkbox("Use AI Topic Labels (GPT-4)", False)
    if use_llm_labeling:
        st.info("""*To use AI Labels:*
                -OpenAI API Key(https://platform.openai.com/api-keys)
                -Billing enabled ($1+ deposit)""")
        openai.api_key = st.text_input("OpenAI API Key", type="password")
    
    # various aspects 
    st.subheader("Aspect Settings")
    aspect_threshold = st.slider("Aspect Similarity Threshold", 0.3, 0.9, 0.65)

#starting the models
@st.cache_resource
def load_models():
    models = {
        "vader": SentimentIntensityAnalyzer(),
        "bert": pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english",
                       framework="pt"),  
        "nlp": spacy.load("en_core_web_md")
    }
    return models

models = load_models()

# Performing the Sentiment Analysis using rhe both models
def analyze_sentiment(text, model_type):
    if model_type == "VADER":
        scores = models["vader"].polarity_scores(text)
        return scores['compound']
    else:  # BERT
        result = models["bert"](text[:512])[0]
        return result['score'] * (1 if result['label'] == 'POSITIVE' else -1)

# Topic labeling using AI
def generate_topic_label(keywords):
    if not use_llm_labeling:
        return " | ".join(keywords[:3])
    
    try:
        client = OpenAI(api_key=openai.api_key)


        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"Generate a concise human-readable label for these keywords: {keywords}. Label:"
            }]
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Labeling Error: {str(e)}")
        return " | ".join(keywords[:3])

# Sentiment defined by the aspect
def aspect_sentiment(text, aspect, nlp):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if aspect.lower() in sent.text.lower()]
    if not sentences: return 0
    return sum(analyze_sentiment(sent, sentiment_model) for sent in sentences) / len(sentences)

if uploaded_file := st.file_uploader("📤 Upload CSV", type=["csv"]):
    df = pd.read_csv(uploaded_file)
    text_col = df.columns[0]

    with st.status("🔍 Processing data...", expanded=True) as status:
        # Cleaning and preprocessing the text from .csv
        df['cleaned'] = df[text_col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        df['processed'] = df['cleaned'].apply(lambda x: " ".join([token.lemma_.lower() 
                                        for token in models["nlp"](x) 
                                        if not token.is_stop and token.is_alpha]))
        
        # Clustering the text
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        X = tfidf.fit_transform(df['processed'])
        kmeans = KMeans(n_clusters=5)
        df['cluster'] = kmeans.fit_predict(X)
        
        # Sentiment Analysis of the text 
        st.write("⚖️ Analyzing sentiment...")
        df['sentiment'] = df[text_col].apply(lambda x: analyze_sentiment(x, sentiment_model))
        
        # Aspect Detection, few predefined, editable in the streamlit
        st.write("🏷️ Detecting aspects...")
        aspects = ["delivery", "quality", "price", "service", "website"]
        for aspect in aspects:
            df[f"{aspect}_sentiment"] = df[text_col].apply(
                lambda x: aspect_sentiment(x, aspect, models["nlp"]))
        
        status.update(label="✅ Processing complete!", state="complete")

    cluster = st.selectbox("Select Cluster", df['cluster'].unique())
    cluster_df = df[df['cluster'] == cluster]

    # visualization of the topic
    with st.expander("📌 Topic Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI-Generated Topic Label")
            keywords = cluster_df['processed'].str.cat(sep=" ").split()[:10]
            topic_label = generate_topic_label(keywords)
            st.markdown(f"## `{topic_label}`")
        
        with col2:
            st.subheader("Topic Keywords")
            wordcloud = WordCloud(width=400, height=200).generate(" ".join(cluster_df['processed']))
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(plt)

    # Dashboard for the Aspect Sentiment 
    st.subheader("📈 Aspect Sentiment Breakdown")
    aspect_scores = {aspect: cluster_df[f"{aspect}_sentiment"].mean() for aspect in aspects}
    aspect_df = pd.DataFrame(list(aspect_scores.items()), columns=["Aspect", "Sentiment"])
    
    chart = alt.Chart(aspect_df).mark_bar().encode(
        x='Aspect',
        y='Sentiment',
        color=alt.condition(
            alt.datum.Sentiment > 0,
            alt.value("green"),
            alt.value("red")
        )
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # Dropdown box for the raw data
    selected_aspect = st.selectbox("🔍 Drill Down into Aspect", aspects)
    aspect_reviews = cluster_df[cluster_df[f"{selected_aspect}_sentiment"].abs() > 0.2]
    
    for _, row in aspect_reviews.head(3).iterrows():
        with st.expander(f"{row[text_col][:50]}..."):
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(row[text_col])
            with col2:
                score = row[f"{selected_aspect}_sentiment"]
                label = "Positive" if score > 0 else "Negative"
                color = "green" if score > 0 else "red"
                st.markdown(f"**{label}** <span style='color:{color};font-size:24px'>{score:.2f}</span>", 
                           unsafe_allow_html=True)
