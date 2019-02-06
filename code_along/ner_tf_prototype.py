import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath('..'))
from pos_baseline_prototype import get_data
from sklearn.utils import shuffle
from util import init_weight
from datetime import datetime
from sklearn.metrics import f1_score

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRnnCell, GRUCell

def get_data(split_sequences=False):
    word2idx = {}
    tag2idx = {}
    word_idx = 1
    tag_idx = 1
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('nlp_class2/ner.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    print("number of samples:", len(Xtrain))
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ntest = int(0.3*len(Xtrain))  # take 30% of the data as test set
    Xtest = Xtrain[:Ntest]
    Ytest = Ytrain[:Ntest]
    Xtrain = Xtrain[Ntest:]
    Ytrain = Ytrain[Ntest:]
    print("number of classes:", len(tag2idx))
    return Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx

def flatten(l):
    return [item for sublist in l for item in sublist]

# get the data
Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data(split_sequences=True)
V = len(word2idx) + 2
K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1

# training config
epochs = 5
learning_rate = 1e-2

if __name__ == '__main__':
