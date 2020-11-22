import json
import glob
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import utils


class TweetObj():
    def __init__(self, obj):
        self.data = obj
        self.text = obj['text']
        self.features = []
        self.is_opinion = obj['opinion']
        self.is_disputed = False
        self.is_political = False

    def feature_extraction(self, ft_engine):
        """
        extract features from text using pre-trained tfidf model
        :param ft_engine: pre-trained tfidf model
        """
        clean_txt = utils.clean_text(self.text)
        self.features = ft_engine.transform([clean_txt]).toarray()
        self.features = self.features.reshape(1, -1)

    def classify_text(self, model):
        """
        Use pre-trained SVC model to identify if the text belongs
        to political topic
        :param model: pre-train SVC model
        """
        ypred = model.predict(self.features)[0]
        cat = utils.get_category_name(ypred)
        self.is_political = (cat == "politics")

    def check_dispute(self):
        """
        Because I don't know how Twitter engine doing this I'm
        randomizing it with equal probability of 1 and 0
        If the text is political and is not an opinion; it's
        subjected to dispute
        """
        self.is_disputed = np.random.randint(0,2)

    def update(self):
        """
        Update tweet object with new parameter
        """
        self.data['is_political'] = self.is_political
        self.data['is_disputed'] = self.is_disputed

    def filter(self, ft_engine, model):
        if not self.is_opinion:
            self.feature_extraction(ft_engine)
            self.classify_text(model)
            if self.is_political:
                self.check_dispute()
        self.update()
