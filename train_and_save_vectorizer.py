import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import json
import numpy as np
import nltk
import re  # Add this import

# Make sure to download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Load intents file
with open('data/intents.json') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # This is where 're' is used
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Prepare training data
patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Convert patterns to features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Encode tags as labels (numerical values)
y = np.array(tags)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Naive Bayes classifier)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'models/chatbot_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")
