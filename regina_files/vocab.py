import pickle as pickle

import numpy as np
from numpy import linalg as LA


class Vocabulary(object):
    def __init__(self, vocab_file, dim_emb=0):
        with open(vocab_file, 'rb') as f:
            self.size, self.word2id, self.id2word = pickle.load(f)
        self.dim_emb = dim_emb
        self.embedding = np.random.random_sample((self.size, self.dim_emb)) - 0.5

        for i in range(self.size):
            self.embedding[i] /= LA.norm(self.embedding[i])
