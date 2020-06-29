import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
#nltk.download('punkt') if not done before

ps = PorterStemmer()

def tokenize(query):
    return nltk.word_tokenize(query)

def stem(w):
    return ps.stem(w.lower())

def bag_of_words(tokenized_query, all_words):
    tq = [stem(w) for w in tokenized_query]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, word in enumerate(all_words):
        if word in tq:
            bag[i] = 1
    return bag