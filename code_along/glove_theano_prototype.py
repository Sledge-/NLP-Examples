import numpy as np
import json
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle

import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from nlp_class2.util import find_analogies

class Glove:
    def __init__(self, D, V, context_sz):
        '''
        V: vocab size
        D: latent feature dimensionality
        context_size: associated words context window size
        '''
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=10e-5, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=True):
        '''
        sentences: weord2index converted sentences
        cc_matrix: path to cc_matrix if it already exists
        learning_rate: learning rate for gradient descent
        reg: regularization factor
        xmax:
        alpha: smoothing factor
        gd: use gradient descent True/False
        use_theano: use theano True/False
        '''

        t0 = datetime.now()
        V = self.V
        D = self.D

        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process:", N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("processed %s/%s" % (it, N))
                n = len(sentence)
                for i in range(n):
                    # i is not word index
                    # j is not the word index
                    wi = sentence[i]  # this is the word index
                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # handle case when context overlaps start/end of sentence
                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi, 0] += points
                        X[0, wi] += points
                    if i + self.context_sz > n:
                        points = 1.0 / (n - i)
                        X[wi, 1] += points
                        X[1, wi] += points

                    # left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j)
                        X[wi, wj] += points
                        X[wj, wi] += points

                    # right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i)
                        X[wi, wj] += points
                        X[wj, wi] += points

            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print( "max value in X:", X.max())

        # weighting and smoothing
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax))**alpha
        fX[X >= xmax] = 1

        print("max in f(X):", fX.max())

        logX = np.log(X + 1)

        print("time to build co-occurence matrix:", (datetime.now() - t0))

        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        if gd and use_theano:
            thW = theano.shared(W)
            thb = theano.shared(b)
            thU = theano.shared(U)
            thc = theano.shared(c)
            thLogX = T.matrix('logX')
            thfX = T.matrix('fX')

            params = [thW, thb, thU, thc]

            thDelta = thW.dot(thU.T) + T.reshape(thb, (V, 1)) + T.reshape(thc, (1,V)) + mu - thLogX
            thCost = (thfX * thDelta * thDelta).sum()

            grads = T.grad(thCost, params)

            updates = [(p, p - learning_rate*g) for p,g in zip(params, grads)]
            train_op = theano.function(
                inputs = [thfX, thLogX],
                updates = updates,
            )

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX # Uses reshape to cast b and c rowwise & columnwise to represent user & movie bias respectively
            cost = (fX * delta * delta). sum()
            costs.append(cost)
            print("epoch:", epoch, "cost:", cost)

            if gd:
                if use_theano:
                    train_op(fX, logX)
                    W = thW.get_value()
                    b = thb.get_value()
                    U = thU.get_value()
                    c = thc.get_value()
                else:
                    oldW = W.copy()
                    for i in range(V):
                        W[i] -= learning_rate*(fX[i,:]*delta[i,:]).dot(U)
                    W -= learning_rate*reg*W

                    for i in range(V):
                        b[i] -= learning_rate*(fX[i,:].dot(delta[i,:]))
                    b -= learning_rate*reg*b

                    for j in range(V):
                        U[j] -= learning_rate*(fX[:,j]*delta[:,j]).dot(oldW)
                    U -= learning_rate*reg*U

                    for j in range(V):
                        c[j] -= learning_rate*fX[:,j].dot(delta[:,j])
                    c -= learning_rate*reg*c
            else:
                #ALS - alternating least squares
                for i in range(V):
                    matrix = reg*np.eye(D) + (fX[i,:]*U.T).dot(U)
                    vector = (fX[i,:]*(logX[i,:] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)

                for i in range(V):
                    denominator = fX[i,:].sum()
                    numerator = fX[i,:].dot(logX[i,:] - W[i].dot(U.T) - c - mu)
                    b[i] = numerator / denominator / (1 + reg)

                for j in range(V):
                    matrix = reg*np.eye(D) + (fX[:,j]*W.T).dot(W)
                    vector = (fX[:,j]*(logX[:,j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)

                for j in range(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W.dot(U[j]) - b - mu)
                    c[j] = numerator / denominator / (1 + reg)


        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)


def main(we_file, w2i_file, n_files=50):
    cc_matrix = "cc_matrix_%s.npy" % n_files

    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(80, V, 10)
    model.fit(sentences, cc_matrix=cc_matrix, epochs=20)
    # model.fit(
    #     sentences=sentences,
    #     cc_matrix=cc_matrix,
    #     learning_rate=3*10e-5,
    #     reg=0.01,
    #     epochs=2000,
    #     gd=True,
    #     use_theano=False,
    # )
    model.save(we_file)


if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    main(we, w2i)

    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w,i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2

        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        # find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)
