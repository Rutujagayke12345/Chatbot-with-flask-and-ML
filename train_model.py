import json
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Define the paths
DATA_FILE = os.path.join("data", "intents.json")
MODEL_FILE = os.path.join("models", "chatbot_model.pkl")

# Load intents from the JSON file
def load_intents(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Preprocess the intents data
def preprocess_intents(intents):
    patterns = []
    tags = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])
    return patterns, tags

def train_and_save_model():
    # Step 1: Load intents
    intents = load_intents(DATA_FILE)
    patterns, tags = preprocess_intents(intents)

    # Step 2: Encode labels
    label_encoder = LabelEncoder()
    encoded_tags = label_encoder.fit_transform(tags)

    # Step 3: Create a machine learning pipeline
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),  # Convert text to feature vectors
        ("classifier", LogisticRegression())  # Train a classifier
    ])

    # Step 4: Train the model
    pipeline.fit(patterns, encoded_tags)

    # Step 5: Save the trained model and label encoder
    os.makedirs("models", exist_ok=True)  # Ensure the models directory exists
    joblib.dump({"model": pipeline, "label_encoder": label_encoder}, MODEL_FILE)
    print(f"Model trained and saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save_model()
