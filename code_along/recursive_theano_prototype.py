import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from sklearn.utils import shuffle
import os
import sys
sys.path.append(os.path.abspath('..'))
from nlp_class2.util import init_weight, display_tree
from get_ptb_data import get_ptb_data
from datetime import datetime

def adagrad(cost, params, lr, eps=1e-10):
    grads = T.grad(cost, params)
    caches = [theano.shared(np.ones_like(p.get_value())) for p in params]
    new_caches = [c + g*g for c, g in zip(caches, grads)]

    c_update = [(c, new_c) for c, new_c in zip(caches, new_caches)]
    g_update = [
      (p, p - lr*g / T.sqrt(new_c + eps)) for p, new_c, g in zip(params, new_caches, grads)
    ]
    updates = c_update + g_update
    return updates

class RecursiveNN:
    def __init__(self, V, D, K):
        self.V = V
        self.D = D
        self.K = K

    def fit(self, trees, learning_rate=3*1e-3, mu=0.99, reg=1e-4, epochs=15, activation=T.nnet.relu, train_inner_nodes=False):
        D = self.D
        V = self.V
        K = self.K
        self.f = activation
        N = len(trees)

        We = init_weight(V, D)
        Wh = np.random.randn(2, D, D) / np.sqrt(2 + D + D)
        bh = np.zeros(D)
        Wo = init_weight(D, K)
        bo = np.zeros(K)

        self.We = theano.shared(We)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wh, self.bh, self.Wo, self.bo]

        words = T.ivector('words')
        parents = T.ivector('parents')
        relations = T.ivector('relations')
        labels = T.ivector('labels')

        def recurrence(n, hiddens, words, parents, relations):
            w = words[n]
            hiddens = T.switch(
                T.ge(w, 0),
                T.set_subtensor(hiddens[n], self.We[w]),
                T.set_subtensor(hiddens[n], self.f(hiddens[n] + self.bh))
            )

            r = relations[n]
            p = parents[n]
            hiddens = T.switch(
                T.ge(p, 0),
                T.set_subtensor(hiddens[p], hiddens[p] + hiddens[n].dot(self.Wh[r])),
                hiddens
            )
            return hiddens

        hiddens = T.zeros((words.shape[0], D))

        h, _ = theano.scan(
            fn=recurrence,
            outputs_info=[hiddens],
            n_steps=words.shape[0],
            sequences=T.arange(words.shape[0]),
            non_sequences=[words, parents, relations],
        )

        py_x = T.nnet.softmax(h[-1].dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)

        rcost = reg*T.mean([(p*p).sum() for p in self.params])
        if train_inner_nodes:
            cost = -T.mean(T.log(py_x[T.arange(labels.shape[0]), labels])) + rcost
        else:
            cost = -T.mean(T.log(py_x[-1, labels[-1]])) + rcost

        # grads = T.grad(cost, self.params)
        # dparams = [theano.shared(p.get_value()*0) for p in self.params]
        #
        # updates = [
        #     (p, p * mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        # ] + [
        #     (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        # ]

        updates = adagrad(cost, self.params, lr=1e-4)

        self.cost_predict_op = theano.function(
            inputs=[words, parents, relations, labels],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        self.train_op = theano.function(
            inputs=[words, parents, relations, labels],
            outputs=[h, cost, prediction],
            updates=updates
        )

        costs = []
        sequence_indexes = range(N)
        if train_inner_nodes:
            n_total = sum(len(words) for words, _, _, _ in trees)
        else:
            n_total = N
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                words, par, rel, lab = trees[j]
                _, c, p  = self.train_op(words, par, rel, lab)
                cost += c
                if train_inner_nodes:
                    n_correct += np.sum(p == lab)
                else:
                    n_correct += (p[-1] == lab[-1])
                it += 1
                if it % 1 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f, cost so far: %f/r" % (it, N, float(n_correct/n_total), cost))
                    sys.stdout.flush()
                print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0))
                costs.append(cost)

        print('costs:', costs)
        plt.plot(costs)
        plt.show()

    def score(self, trees):
        n_total = len(trees)
        n_correct = 0
        for words, par, rel, lab in trees:
            _, p = self.cost_predict_op(words, par, rel, lab)
            n_correct += (p[-1] == lab[-1])
        return float(n_correct) / n_total


# DATA PROCESSING FUNCTIONS:
def add_idx_to_tree(tree, current_idx):
    if tree is None:
        return current_idx
    current_idx = add_idx_to_tree(tree.left, current_idx)
    current_idx = add_idx_to_tree(tree.right, current_idx)
    tree.idx = current_idx
    current_idx += 1
    return current_idx

def tree2list(tree, parent_idx, is_binary=False, is_left=False, is_right=False):
    if tree is None:
        return [], [], [], []
    w = tree.word if tree.word is not None else -1
    if is_left:
        r = 0
    elif is_right:
        r = 1
    else:
        r = -1
    words_left, parents_left, relations_left, labels_left = tree2list(tree.left, tree.idx, is_binary, is_left=True)
    words_right, parents_right, relations_right, labels_right = tree2list(tree.right, tree.idx, is_binary, is_right=True)

    words = words_left + words_right + [w]
    parents = parents_left + parents_right + [parent_idx]
    relations = relations_left + relations_right + [r]
    if is_binary:
        if tree.label > 2:
            label = 1
        elif tree.label < 2:
            label = 0
        else:
            label = -1
    else:
        label = tree.label
    labels = labels_left + labels_right + [label]
    return words, parents, relations, labels

def main(is_binary=True):
    train, test, word2idx = get_ptb_data()

    for t in train:
        add_idx_to_tree(t, 0)
    train = [tree2list(t, -1, is_binary) for t in train]
    if is_binary:
        train = [t for t in train if t[3][-1] >= 0]

    for t in test:
        add_idx_to_tree(t, 0)
    test = [tree2list(t, -1, is_binary) for t in test]
    if is_binary:
        test = [t for t in test if t[3][-1] >= 0]

    train = shuffle(train)
    train = train[:2000]
    test = shuffle(test)
    test = test[:100]

    V = len(word2idx)
    D = 10
    K = 2 if is_binary else 5

    model = RecursiveNN(V, D, K)
    model.fit(train, learning_rate=10e-3, reg=10e-3, mu=0, epochs=30, activation=T.tanh, train_inner_nodes=False)
    print("train accuracy:", model.score(train))
    print("test accuracy:", model.score(test))




if __name__ =='__main__':

    main()
