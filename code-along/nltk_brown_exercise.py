# Load the Brown corpus using NLTK
# Create a bigram language model with add-one smoothing
# basically just count up unigrams and divide by bigram where the unigram is the 2nd term
# downcase the tokens
# test the probability of a real sentence vs. fake sentence.

import nltk
from nltk.corpus import brown
print(brown.categories())
