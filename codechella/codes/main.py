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
from classify_text import TweetObj
import sys
from fetchTweets import *
import argparse


def print_result(obj, tfidf, model):
    data_obj = TweetObj(obj)
    data_obj.filter(tfidf, model)

    ### display message
    print("Tweet:", data_obj.text)
    if data_obj.is_political:
        if not data_obj.is_opinion and data_obj.is_disputed:
            print("Notification: This tweet involves political topic which was tagged to be disputed and was not confirmed by the author to be an opinion.\n Twitter will hide this claim until information can be verified")
        if data_obj.is_opinion:
            print("This is an opinion")



def main(args_to_parse):
    parser = argparse.ArgumentParser(description="Opinion tag arguments.")
    parser.add_argument("-w", "--write", help="If you want to enter your own tweet, follow your tweet in a quotation")
    parser.add_argument("-ot", "--opiniontag", help="True (1) or False (0) if you claim your tweet is an opinion")
    parser.add_argument("-r", "--read", help="Path to JSON file that contains tweets you want to read")
    parser.add_argument("-dl", "--download", help="Twitter handle of the account you want to download tweets from")

    args = parser.parse_args(args_to_parse)
    is_write = bool(args.write)
    if is_write:
        data = args.write
    is_opinion = args.opiniontag

    is_read = bool(args.read)
    if is_read:
        filepath = args.read

    is_download = bool(args.download)
    if is_download:
        userhandle = args.download

    if int(is_write + is_read + is_download) != 1:
        parser.error("No input provided, add --write, --read or --download")
    elif 2 <= int(is_write + is_read + is_download):
        parser.error("You can only choose one among --write, --read and --download mode")
    if is_write and not is_opinion:
        parser.error("--write mode needs to be followed by --opiniontag")

    ### Load pre-trained model for filtering
    with open('../models/tfidf.pickle', 'rb') as file:
        TFIDF = pickle.load(file)
    with open("../models/best_svc.pickle", 'rb') as file:
        MODEL = pickle.load(file)

    ### loading data
    if is_write:
        data = {'text': data, 'opinion': bool(int(is_opinion))}
    elif is_read:
        data = utils.load_tweet_object(filepath)
        for d in data:
            d['opinion'] = False
    elif is_download:
        data = pull_tweets(userhandle, 20)
        for d in data:
            d['opinion'] = False

    ### print result of filtering process
    if is_write:
        print_result(data, TFIDF, MODEL)
    else:
        for d in data:
            print_result(d, TFIDF, MODEL)


if __name__ == "__main__":
    #args = parse_arguments(sys.args[1:])
    main(sys.argv[1:])
