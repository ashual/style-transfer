from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np


class WordIndexingEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, datasets, embedding_size, n=1, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self, save_dir, datasets, n, truncate_by_cutoff)
        self.embedding_size = embedding_size
        self.load_or_create()

    def create_embedding(self, vocab):
        return np.random.rand(len(vocab), self.embedding_size)
