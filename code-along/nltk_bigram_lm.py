# Load the Brown corpus using NLTK
# Create a bigram language model with add-one smoothing
# downcase the tokens, strip punctuation
# test the probability of a real sentence vs. fake sentence.

import math
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from collections import Counter

class BigramLanguageModel:
    def __init__(self):
        self.uni_corpus = []
        self.bigram_corpus = []
        self.unigram_cnt = {}
        self.bigram_cnt = {}
        self.V = 0
        self.get_corpus()
        self.get_counts()

    def get_corpus(self):
        for fileid in brown.fileids():
            sentences = brown.sents(fileids=[fileid])
            for s in sentences:
                clean_sentence = [w.lower() for w in s if w.isalpha()]
                self.uni_corpus.extend(clean_sentence)
                self.bigram_corpus.extend(list(nltk.bigrams(clean_sentence)))

    def get_counts(self):
        self.unigram_cnt = dict(Counter(self.uni_corpus))
        self.bigram_cnt = dict(Counter(self.bigram_corpus))
        self.V = len(self.unigram_cnt.keys())
        print("number of unique words in corpus: ", self.V)

    def get_log_prob(self, w1, w2):
        '''
        Return log probibility of w2 given w1.
        P(w1 -> w2) = cnt(w1 -> w2)/cnt(w2)
        '''
        bigram = (w1, w2)
        try:
            prob = (self.bigram_cnt[bigram] + 1)/(self.unigram_cnt[w1] + self.V)  # uses add 1 smoothing
        except KeyError:
            return float("-inf")
        return math.log(prob)

    def score_sentence(self, sent):
        sent = word_tokenize(sent)
        print(sent)
        bigrams = list(nltk.bigrams(sent))
        log_prob = 0
        for bigram in bigrams:
            w1, w2 = bigram
            log_prob += self.get_log_prob(w1, w2)
        log_prob /= len(sent) + 1       # normalize the score
        return log_prob

if __name__ == '__main__':
    bglm = BigramLanguageModel()
    print(bglm.get_log_prob('the', 'fulton'))
    print(bglm.get_log_prob('the', 'dog'))
    print(bglm.get_log_prob('the', 'xfd'))

    print(bglm.get_log_prob('the', 'quick'))
    print(bglm.get_log_prob('quick', 'brown'))
    print(bglm.get_log_prob('brown', 'fox'))
    print(bglm.score_sentence('the quick brown fox'))

    print(bglm.get_log_prob('the', 'fulton'))
    print(bglm.get_log_prob('fulton', 'county'))
    print(bglm.get_log_prob('county', 'grand'))
    print(bglm.get_log_prob('grand', 'jury'))

    log_prob = bglm.score_sentence('the fulton county grand jury')
    print(log_prob)
    print(math.exp(log_prob))

    log_prob = bglm.score_sentence('he was born on a farm')
    print(log_prob)
    print(math.exp(log_prob))

    log_prob = bglm.score_sentence('this relatively small species has distinct pale blue eyes')
    print(log_prob)
    print(math.exp(log_prob))
