# scripts/intent_recognition.py

import joblib
import numpy as np
import re
from nltk.stem import WordNetLemmatizer

# Load the model and vectorizer
model = joblib.load('models/chatbot_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text (split by spaces)
    words = text.split()

    # Lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def predict_intent(cleaned_input):
    # Transform the cleaned input to vector format
    vectorized_input = vectorizer.transform([cleaned_input])

    # Predict the intent using the trained model
    intent = model.predict(vectorized_input)

    return intent[0]  # Return the predicted intent
