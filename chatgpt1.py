import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import logging
from wordcloud import WordCloud

# Suppress TensorFlow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Streamlit UI setup
st.title("ChatGPT Reviews Sentiment Analysis")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Set the background image
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg_image("https://img.freepik.com/free-vector/gray-white-background-with-glowing-light-effect-design_1017-40446.jpg?w=2000")

# Load pre-trained model
model_path = "sentiment_model.h5"
model = None
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    st.sidebar.success("Pre-trained Model Loaded Successfully!")
else:
    st.sidebar.error("Pre-trained Model Not Found. Please Upload it.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    st.write("### Dataset Preview:")
    st.write(df.head())

    if 'review' not in df.columns or 'rating' not in df.columns:
        st.error("Dataset must contain 'review' and 'rating' columns.")
    else:
        df['cleaned_review'] = df['review'].fillna("")
        df['sentiment'] = df['rating'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})

        max_words = 10000
        max_length = 200
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(df['cleaned_review'])

        user_input = st.text_area("Enter a review to predict sentiment:")
        if st.button("Predict Sentiment") and model is not None:
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_length)
            prediction = model.predict(padded)
            sentiment_classes = ['Negative', 'Neutral', 'Positive']
            sentiment = sentiment_classes[np.argmax(prediction)]
            st.write(f"Predicted Sentiment: **{sentiment}**")

        st.write("## Sentiment Analysis Insights")
        analysis_option = st.selectbox("Select a question to explore:", [
            "1. What is the overall sentiment of user reviews?",
            "2. How does sentiment vary by rating?",
            "3. Which keywords or phrases are most associated with each sentiment class?",
            "4. How has sentiment changed over time?",
            "5. Do verified users tend to leave more positive or negative reviews?",
            "6. Are longer reviews more likely to be negative or positive?",
            "7. Which locations show the most positive or negative sentiment?",
            "8. Is there a difference in sentiment across platforms (Web vs Mobile)?",
            "9. Which ChatGPT versions are associated with higher/lower sentiment?",
            "10. What are the most common negative feedback themes?"
        ])

        def plot_custom_bar(data, title, xlabel, ylabel, horizontal=False):
            fig, ax = plt.subplots(figsize=(10, 5))
            if isinstance(data, pd.Series):
                data = data.sort_values()
                if horizontal:
                    data.plot(kind='barh', color='skyblue', ax=ax)
                else:
                    data.plot(kind='bar', color='skyblue', ax=ax)
            else:
                data = data.fillna(0)
                if horizontal:
                    data.plot(kind='barh', stacked=True, ax=ax)
                else:
                    data.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            st.pyplot(fig)

        if analysis_option == "1. What is the overall sentiment of user reviews?":
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            ax.axis('equal')
            st.pyplot(fig)

        elif analysis_option == "2. How does sentiment vary by rating?":
            sentiment_by_rating = df.groupby('rating')['sentiment'].value_counts().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(sentiment_by_rating, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            ax.set_title("Sentiment Distribution by Rating")
            st.pyplot(fig)

        elif analysis_option == "3. Which keywords or phrases are most associated with each sentiment class?":
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_review'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
                st.write(f"#### {sentiment} Reviews Word Cloud")
                st.image(wordcloud.to_array())

        elif analysis_option == "4. How has sentiment changed over time?":
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                sentiment_trend = df.groupby(df['date'].dt.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)
                st.line_chart(sentiment_trend)
            else:
                st.warning("Date column not available in the dataset.")

        elif analysis_option == "5. Do verified users tend to leave more positive or negative reviews?":
            if 'verified_purchase' in df.columns:
                sentiment_by_verified = df.groupby('verified_purchase')['sentiment'].value_counts().unstack().fillna(0)
                plot_custom_bar(sentiment_by_verified, "Sentiment by Verified Purchase", "Verified Purchase", "Count")
            else:
                st.warning("verified_purchase column not available in the dataset.")

        elif analysis_option == "6. Are longer reviews more likely to be negative or positive?":
            df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x='sentiment', y='review_length', palette="Set2", ax=ax)
            ax.set_title("Review Length Distribution by Sentiment")
            st.pyplot(fig)

        elif analysis_option == "7. Which locations show the most positive or negative sentiment?":
            if 'location' in df.columns:
                sentiment_by_location = df.groupby('location')['sentiment'].value_counts().unstack().fillna(0)
                plot_custom_bar(sentiment_by_location, "Sentiment by Location", "Location", "Count")
            else:
                st.warning("Location column not available in the dataset.")

        elif analysis_option == "8. Is there a difference in sentiment across platforms (Web vs Mobile)?":
            if 'platform' in df.columns:
                sentiment_by_platform = df.groupby('platform')['sentiment'].value_counts().unstack().fillna(0)
                plot_custom_bar(sentiment_by_platform, "Sentiment by Platform", "Platform", "Count")
            else:
                st.warning("Platform column not available in the dataset.")

        elif analysis_option == "9. Which ChatGPT versions are associated with higher/lower sentiment?":
            if 'version' in df.columns:
                sentiment_by_version = df.groupby('version')['sentiment'].value_counts().unstack().fillna(0)
                plot_custom_bar(sentiment_by_version, "Sentiment by ChatGPT Version", "Version", "Count")
            else:
                st.warning("Version column not available in the dataset.")

        elif analysis_option == "10. What are the most common negative feedback themes?":
            negative_words = ' '.join(df[df['sentiment'] == 'Negative']['cleaned_review'])
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words)
            st.image(wordcloud_negative.to_array())
