# build a logistic regression model to estimate the bigram probabilities
# TODO: Fix the out of memory error.

import math
import nltk
from nltk.corpus import brown
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class LogregBigramLanguageModel:
    def __init__(self):
        self.uni_corpus = []
        self.bigram_corpus = []
        self.uni_one_hot = []
        self.bigram_one_hot_X = []
        self.bigram_one_hot_Y = []
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.lr = LogisticRegression()
        self.get_corpus()
        self.fit_one_hot()
        self.train_model(self.bigram_one_hot_X, self.bigram_one_hot_Y)
        # self.test_model()

    def get_corpus(self):
        for s in brown.sents():
            clean_sentence = [w.lower() for w in s if w.isalpha()]
            self.uni_corpus.extend(clean_sentence)
            self.bigram_corpus.extend(list(nltk.bigrams(clean_sentence)))

    def fit_one_hot(self):
        self.enc.fit([[w] for w in self.uni_corpus])
        self.uni_one_hot = self.enc.transform([[w] for w in self.uni_corpus])
        w1s, w2s = zip(*self.bigram_corpus)
        self.bigram_one_hot_X = self.enc.transform([[w] for w in w1s])
        self.bigram_one_hot_Y = self.enc.transform([[w] for w in w2s])

    def train_model(self, X, Y):
        self.lr.fit(X, Y)

    # def test_model(self):
    #     # TODO: Fix the out of memory error.
    #     X_train, X_test, Y_train, Y_test = train_test_split(self.bigram_one_hot_X, self.bigram_one_hot_Y, test_size = 0.1)
    #     self.train_model(X_train, Y_train)
    #     Y_pred = self.lr.predict(X_test)
    #     print(confusion_matrix(Y_test, Y_pred))
    #     print(classification_report(Y_test, Y_pred))

    def get_prob(self, w1, w2):
        w1_one_hot = self.enc.transform([[w1]])
        w2_one_hot = self.enc.transform([[w2]])
        print(w1_one_hot)
        print(w2_one_hot)



if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    logreg_bigram_lm = LogregBigramLanguageModel()
    print(logreg_bigram_lm.bigram_one_hot_X)

    logreg_bigram_lm.get_prob('the', 'fox')
