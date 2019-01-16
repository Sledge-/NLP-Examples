# build a logistic regression model to estimate the bigram probabilities
import math
import nltk
from nltk.corpus import brown
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

class LogregBigramLanguageModel:
    def __init__(self):
        self.uni_corpus = []
        self.bigram_corpus = []
        self.uni_one_hot = []
        self.bigram_one_hot_X = []
        self.bigram_one_hot_Y = []
        self.get_corpus()
        self.one_hot()

    def get_corpus(self):
        for s in brown.sents():
            clean_sentence = [w.lower() for w in s if w.isalpha()]
            self.uni_corpus.extend(clean_sentence)
            self.bigram_corpus.extend(list(nltk.bigrams(clean_sentence)))

    def one_hot(self):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(self.unii_corpus)


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
