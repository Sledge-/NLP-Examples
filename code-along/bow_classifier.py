from __future__ import print_function, division
from builtins import range

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors

train = pd.read_csv('../umb_textmining/r8-train-all-terms.txt', header=None, sep='\t')
test = pd.read_csv('../umb_textmining/r8-test-all-terms.txt', header=None, sep='\t')
train.columns = ['label', 'content']
test.columns = ['label', 'content']


class GloveVectorizer:
    def __init__(self):
        print('Loading word vectors...')
        word2vec = {}
