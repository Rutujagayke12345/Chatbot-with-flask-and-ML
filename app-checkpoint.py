from flask import Flask, render_template, request
import json
import nltk
from scripts.utils import load_intents, clean_text
from scripts.intent_recognition import predict_intent

# Initialize the Flask application
app = Flask(__name__)

# Load the intents file and model
intents = load_intents('data/intents.json')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form['user_input']
    cleaned_input = clean_text(user_input)
    intent = predict_intent(cleaned_input)

    # Find the response based on the predicted intent
    response = "I'm sorry, I don't understand that."
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            response = intent_data['responses'][0]
            break

    return response

if __name__ == "__main__":
    app.run(debug=True)
