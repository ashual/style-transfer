from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np
from os import getcwd
from os.path import join


class GloveEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, pretrained_glove_file=None, dataset=None, n=1, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self, save_dir, dataset, n, truncate_by_cutoff)
        if pretrained_glove_file is None:
            self.pretrained_glove_file = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")
        else:
            self.pretrained_glove_file = pretrained_glove_file
        self.load_or_create()

    @staticmethod
    def create_random_token():
        '''
        Create random array with numbers between -maximum_value to maximum_value
        :return:
        '''
        embedding_size = 50
        maximum_value = 5.
        return (np.random.random(size=embedding_size) - 0.5) * (2. * maximum_value)

    def load_glove(self, vocab):
        glove_dict = {}
        embedding = []
        file = open(self.pretrained_glove_file, 'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            glove_dict[row[0]] = row[1:]

        for word in vocab:
            if word in glove_dict:
                embedding.append(glove_dict[word])
            else:
                embedding.append(self.create_random_token())
        print('Loaded GloVe!')
        file.close()
        return embedding

    def create_embedding(self, vocab):
        embedding_list = self.load_glove(vocab)
        return np.asarray(embedding_list, dtype=np.float32)
