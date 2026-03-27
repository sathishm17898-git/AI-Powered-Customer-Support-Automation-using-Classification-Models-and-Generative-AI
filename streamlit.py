import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
#For Deep learning
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import streamlit as st
import pickle, joblib
from tensorflow.keras.models import load_model

model = load_model("D:/Sathish/AIML/Fianl Project/ticket_classifier.h5")
with open("D:/Sathish/AIML/Fianl Project/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
scaler = joblib.load("D:/Sathish/AIML/Fianl Project/scaler.pkl")
with open("D:/Sathish/AIML/Fianl Project/le.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("AI Support assistant")
st.write("I am your assistant, how can i help you")
user_input = st.text_area("Enter your question:")

def clean_text(user_input):
    user_input = user_input.lower()
    user_input = re.sub("'", "", user_input)
    user_input = re.sub(r'\s+', ' ', user_input)
    user_input = re.sub(r'\n+', ' ', user_input)
    tokens = word_tokenize(user_input, language='english')
    token = [i for i in tokens if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(token)

keywords = ["refund", "login issue", "payment failed"]
urgency_words = ["urgent","asap","immediately"]
sia = SentimentIntensityAnalyzer()
def keyword_features(text):
    return {f"has_{kw}": (kw in text.lower()) for kw in keywords}
def urgency_features(text):
    return {f"has_{ug}": (ug in text.lower()) for ug in urgency_words}
def sentiment_score(text):
    scores = sia.polarity_scores(text)
    return {
        "pos": scores['pos'],
        "neg": scores['neg'],
        "neu": scores['neu'],
        "com": scores['compound']
    }
def extract_features(text, scaler=None):
    features = {}
    features.update(keyword_features(text))
    features.update(urgency_features(text))
    features.update(sentiment_score(text))

    # Convert dict to array in fixed order
    feat_array = np.array([
        features["has_refund"],
        features["has_login issue"],
        features["has_payment failed"],
        features["has_urgent"],
        features["has_asap"],
        features["has_immediately"],
        features["pos"],
        features["neg"],
        features["neu"],
        features["com"]
    ]).reshape(1, -1)

    # Scale only pos, neg, neu (columns 6,7,8)
    if scaler is not None:
        raw_part = feat_array[:, :6]          # keyword + urgency
        scaled_part = scaler.transform(feat_array[:, 6:])  # pos, neg, neu      # com left unscaled
        feat_array = np.concatenate([raw_part, scaled_part], axis=1)

    return feat_array

if st.button("Submit"):
    # Text preprocessing
    seq = tokenizer.texts_to_sequences([user_input])
    pad = pad_sequences(seq, maxlen=100)

    # Numeric feature extraction
    num_features = extract_features(user_input, scaler)

    # Predict
    pred = model.predict([pad, num_features])
    class_id = np.argmax(pred, axis=1)[0]
    # Convert integer back to original category
    predicted_class = label_encoder.inverse_transform([class_id])[0]

    st.write(f"Your message is received by the {predicted_class} department")

    import google.generativeai as genai

    # Configure Gemini
    genai.configure(api_key="AIzaSyDrCGzTo9L99YO2BmqbXRI1RLOiU8xXb2Y")
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

    # After prediction
    pred = model.predict([pad, num_features])
    class_id = np.argmax(pred, axis=1)[0]
    predicted_class = label_encoder.inverse_transform([class_id])[0]

    # Gemini polite response
    response = gemini_model.generate_content(
    f"You are a customer support assistant. "
    f"The following ticket has been classified into the '{predicted_class}' queue. "
    f"Ticket text: '{user_input}'. "
    f"Please generate a short, polite acknowledgment (2 to 3 sentences) "
    f"that reassures the customer their issue is being handled."
)

    st.write(f"Support Assistant: {response.text}")




