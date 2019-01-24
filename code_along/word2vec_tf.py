# Outline:
# 1. Load the data.
# 2. Build and train the Model.
# 3. Test the trained model (find analogies).

from __future__ import print_function, division
from builtins import range

import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
# from util import find_analogies

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

from glob import glob

import os
import sys
import string
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-files", type=int, default=None,
	help="number of files to use for model training.")
ap.add_argument("-r", "--randomize-files", type=str, default="n",
	help="randomize files Y/n.")
args = vars(ap.parse_args())
num_files = args['num_files']
randomize_files = True if args['randomize_files'] in ['Y','Yes','y','yes'] else False


sys.path.append(os.path.abspath('..'))
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab as get_brown

def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('', '', string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

def get_wiki():
    '''
    Loop through the vocab.  Take the top V words.  Build
    the word2idx dict.  Return the indexed sentences and
    the word2idx dict.
    '''
    V = 20000
    files = glob('../wikimedia/enwiki*.txt')
    print(type(files))
    if randomize_files:
        random.shuffle(files)
    all_word_counts = {}
    # 1st loop through, count the words and build the
    # word2index dictionary
    print("%s wiki files found, using %s of them" % (len(files), "all" if num_files is None else num_files))
    if num_files is not None:
        files = files[:num_files]
    print("loading and counting wiki data")
    for f in files:
        print("loading: %s" % f)
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}]':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1
    print("finished counting")

    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    # 2nd loop through, use the word2idx dictionary to build
    # indexed sentences if sentences is > 1 length
    sents = []
    print("indexing sentences")
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}]':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sent = [word2idx[w] if w in word2idx else unk for w in s]
                    sents.append(sent)
    return sents, word2idx

def train_model(savedir):
    '''
    Loop through the training sentences and perform SGD
    using negative sampling and ...
    '''
    sentences, word2idx = get_wiki()

    vocab_size = len(word2idx)

    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5 # negative samples per input word
    epochs = 20
    D = 50 # word embedding dimension size

    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    W = np.random.randn(vocab_size, D).astype(np.float32) # input-to-hidden
    V = np.random.randn(D, vocab_size).astype(np.float32) # hidden-to-output

    tf_input = tf.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.placeholder(tf.int32, shape=(None,))
    tf_context = tf.placeholder(tf.int32, shape=(None,))
    tfW = tf.Variable(W)
    tfV = tf.Variable(V.T)

    def dot(A, B):
        C = A * B
        return tf.reduce_sum(C, axis=1)

    # correct middle word output
    emb_input = tf.nn.embedding_lookup(tfW, tf_input)
    emb_output = tf.nn.embedding_lookup(tfV, tf_context)
    correct_output = dot(emb_input, emb_output)
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones(tf.shape(correct_output)), logits=correct_output)

    emb_input = tf.nn.embedding_lookup(tfW, tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros(tf.shape(incorrect_output)), logits=incorrect_output)

    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)

    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)

    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    p_neg = get_negative_sampling_distribution(sentences, vocab_size)

    costs = []

    total_words = sum(len(sentence) for sentence in sentences)
    print("total number of words in corpus:", total_words)

    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    # TRAINING LOOP:
    for epoch in range(epochs):
        np.random.shuffle(sentences)

        cost = 0
        counter = 0
        inputs = []
        targets = []
        negwords = []
        t0 = datetime.now()
        for sentence in sentences:
            sentence = [w for w in sentence if np.random.random() < (1 - p_drop[w])]
            if len(sentence) < 2:
                continue

            randomly_ordered_positions = np.random.choice(
                len(sentence),
                size=len(sentence),
                replace=False
            )

            for j, pos in enumerate(randomly_ordered_positions):
                word = sentence[pos]

                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p=p_neg)

                n = len(context_words)
                inputs += [word]*n
                negwords += [neg_word]*n
                targets += context_words

            if len(inputs) >= 128:
                _, c = session.run(
                    (train_op, loss),
                    feed_dict={
                        tf_input: inputs,
                        tf_negword: negwords,
                        tf_context: targets,
                    }
                )
                cost += c

                inputs = []
                targets = []
                negwords = []

            counter += 1
            tot = len(sentences)
            if counter % 1000 == 0:
                time_remaining = int(est_time_remaining(t0, datetime.now(), counter, tot))
                m, s = divmod(time_remaining, 60)
                h, m = divmod(m, 60)
                sys.stdout.write("processed %s /%s, estimate %d:%02d:%02d remaining\r" % (counter, tot, h, m, s))
                sys.stdout.flush()

        dt = datetime.now() - t0
        print("epoch complete:", epoch, "cost:", cost, "dt:", dt)

        costs.append(cost)

        learning_rate -= learning_rate_delta

    plt.plot(costs)
    plt.show()

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)

    np.savez('%s/weights.npz' % savedir, W, V)

    return word2idx, W, V

def get_negative_sampling_distribution(sentences, vocab_size):
    word_freq = np.zeros(vocab_size)
    word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1

    # add smoothing
    p_neg = word_freq**0.75

    # add normalization
    p_neg = p_neg / p_neg.sum()

    assert(np.all(p_neg > 0))
    return p_neg

def get_context(pos, sentence, window_size):
    # input:
    # a sentence of the form: x x x c c c pos c c c x x x
    # output:
    # the context word indices: c c c c c c

    start = max(0, pos - window_size)
    end_ = min(len(sentence), pos + window_size)

    context = []
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start):
        if ctx_pos != pos:
            context.append(ctx_word_idx)
    return context

def sgd(input_, targets, label, learning_rate, W, V):
    activation = W[input_].dot(V[:,targets])
    prob = sigmoid(activation)

    # gradients
    gV = np.outer(W[input_], prob - label)  # D x N
    gW = np.sum((prob - label) * V[:,targets], axis=1) # D

    V[:, targets] -= learning_rate * gV # D x N
    W[input_] -= learning_rate * gW # D

    cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
    return cost.sum()

def load_model(savedir):
    with open('%s/word2idx.json' % savedir) as f:
        word2idx = json.load(f)
    npz = np.load('%s/weights.npz' % savedir)
    W = npz['arr_0']
    V = npz['arr_1']
    return word2idx, W, V

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape

    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("sorry, %s not in word2idx" % w)
            return

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1,D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]

    for i in idx:
        if i not in keep_out:
            best_idx = i
            break

    print("got %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])
    print("dist to %s" % pos2, cos_dist(p2, vec))

def test_model(word2idx, W, V):
    idx2word = {i:w for w, i in word2idx.items()}
    for We in (W, (W+ V.T) / 2):
         print("***************")

         analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
         analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
         analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
         analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
         analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
         analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
         analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
         analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
         analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
         analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
         analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
         analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
         analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
         analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
         analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
         analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
         analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
         analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
         analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
         analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
         analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
         analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
         analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
         analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
         analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
         analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
         analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)

def est_time_remaining(t0, tn, n, N):
    elapsed_time = (tn - t0).total_seconds()
    pct_complete = n/N
    est_finish_time = elapsed_time / (pct_complete)
    est_time_remaining = est_finish_time - elapsed_time
    return est_time_remaining

if __name__ == '__main__':
    word2idx, W, V = train_model('w2v_model')
    test_model(word2idx, W, V)




#
