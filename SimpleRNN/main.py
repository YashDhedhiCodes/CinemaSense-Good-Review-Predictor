# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Text Preprocessing with Symbol Cleaning
def preprocess_text(text):
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = cleaned.split()
    encoded_review = []

    for word in words:
        index = word_index.get(word, 2) + 3
        if index < 10000:  #Keep only allowed words for the embedding layer
            encoded_review.append(index)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return np.array(padded_review)


# Step 3: Streamlit UI Styling
st.set_page_config(page_title="CinemaSense - Review Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea > div > textarea {
        background-color: #fff !important;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        border: none;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

# Title and Description
st.title('🍿 CinemaSense: Movie Review Sentiment Classifier')
st.write("🎬 Enter a movie review below and get a prediction on whether it's **Positive** or **Negative**.")

# User input
user_input = st.text_area('📝 Your Movie Review:', height=150)

# Predict button
if st.button('🚀 Classify Review'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before clicking classify.")
    else:
        preprocessed_input = preprocess_text(user_input)
        if preprocessed_input is not None and preprocessed_input.shape == (1, 500):
            prediction = model.predict(preprocessed_input)
            sentiment = '🟢 Positive' if prediction[0][0] > 0.5 else '🔴 Negative'
            st.success(f'**Sentiment:** {sentiment}')
            st.info(f'**Prediction Confidence:** {prediction[0][0]:.2f}')
        else:
            st.error("⚠️ Invalid input format or empty review.")
else:
    st.caption("👆 Type your review and click the button to classify.")

st.markdown('</div>', unsafe_allow_html=True)
