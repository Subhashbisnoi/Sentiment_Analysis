import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Recreate the TextPreprocessor class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.remove('not')

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        corpus = []
        for review in X:
            review = re.sub(r'[^a-zA-Z]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.ps.stem(word) for word in review if word not in self.stopwords]
            corpus.append(' '.join(review))
        return corpus

# Load the trained pipeline
model_path = 'sentiment_analysis_pipeline.pkl'
pipeline = pickle.load(open(model_path, 'rb'))

# Streamlit UI
st.title("🍽️ Restaurant Review Sentiment Analyzer")

st.write("Enter a restaurant review below, and we'll analyze whether it's **positive** or **negative**.")

# User input
user_review = st.text_area("✍️ Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_review.strip():  # Ensure the user entered something
        # Predict sentiment
        prediction = pipeline.predict([user_review])[0]
        
        # Display result
        if prediction == 1:
            st.success("✅ **Positive Review** 😊")
        else:
            st.error("❌ **Negative Review** 😞")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# Footer
st.write("🚀 Built with ❤️ using Streamlit")