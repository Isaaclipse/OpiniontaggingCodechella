import json
import glob
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_tweet_object(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.read())
    return data

def lemmatized(text):
    """
    Iterate through every word to lemmatize text
    :param text: string of words to lemmatize
    """
    ### Download package if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/wordnet')
    except:
        nltk.download('wordnet')

    wordnet_lemmatizer = WordNetLemmatizer()
    lem = []
    words = text.split(" ")
    for word in words:
        lem.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    return " ".join(lem)

def clean_text(text):
    """
    :param text: string of words
    """
    punc = list("?:!.,;")
    stop_words = list(stopwords.words('english'))
    data = text
    lemmatized_list = []
    ### clean text
    data = data.replace("\r", " ")
    data = data.replace("\n", " ")
    data = data.replace("    ", " ")
    data = data.replace('"', '')  # quoted text
    data = data.lower()
    for p in punc:
        data = data.replace(p, "")
    data = data.replace("'s", "")
    data = lemmatized(data)
    for sw in stop_words:
        regex_sw = r"\b" + sw + r"\b"
        data = data.replace(regex_sw, "")
    return data

def get_category_name(category_id):
    """
    Get topic category of text
    :param category_id: id of the category
    """
    cat_codes = {
        0: 'business',
        1: 'entertainment',
        2: 'politics',
        3: 'sport',
        4: 'tech'
        }
    return cat_codes[category_id]
