import json
import nltk
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def clean_text(text):
    # Remove punctuation and convert text to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    return text

def tokenize_text(text):
    return nltk.word_tokenize(text)
