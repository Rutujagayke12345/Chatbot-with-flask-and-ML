from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import numpy as np
from scripts.intent_recognition import predict_intent, preprocess_text

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/chatbot_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        user_input = request.json.get("user_input")  # Get input from the AJAX request
        cleaned_input = preprocess_text(user_input)  # Preprocess input
        vectorized_input = vectorizer.transform([user_input])  # Vectorize input
        intent = predict_intent(cleaned_input)  # Predict intent

        if intent:
            response = generate_response(intent)  # Generate a response for the predicted intent
        else:
            response = "Sorry, I didn't understand that."

        return jsonify({'response': response})  # Send the response back as JSON

    return render_template("index.html")

def generate_response(intent):
    # Based on the intent, return a predefined response or any dynamic logic you have
    responses = {
        'greeting': "Hello! How can I assist you?",
        'goodbye': "Goodbye! Have a great day!",
        'thanks': "You're welcome! Glad I could help."
    }
    return responses.get(intent, "I'm not sure how to respond to that.")

if __name__ == "__main__":
    app.run(debug=True)
