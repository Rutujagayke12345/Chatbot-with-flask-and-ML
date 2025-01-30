import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Tokenizes and lemmatizes the input text.
    """
    words = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in words]
