from ner_baseline_prototype import get_data
from pos_rnn_prototype import RNN

def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data(split_sequences=True)
    V = len(word2idx)
    K = len(tag2idx)
    rnn = RNN(10, [10], V, K) # using RNN from POS tagger because semantically it's the same analysis
    rnn.fit(Xtrain, Ytrain, epochs=100)
    print("train score:", rnn.score[Xtrain, Ytrain])
    print("train f1 score:", rnn.f1_score[Xtrain, Ytrain])
    print("test score:", rnn.score[Xtest, Ytest])
    print("test f1 score:", rnn.f1_score[Xtest, Ytest])

if __name__ == "__main__":
    main()
