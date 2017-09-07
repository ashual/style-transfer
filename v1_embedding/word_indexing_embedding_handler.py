from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np


class WordIndexingEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, datasets, embedding_size, n=1, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self, save_dir)
        if not self.initialized_from_cache:
            print('creating embedding...')
            vocab = self.build_dataset(datasets, n, truncate_by_cutoff)
            # init mappings
            self.vocabulary_to_internals(vocab)
            # init embeddings
            self.embedding_np = np.random.rand(len(vocab), embedding_size)
            self.save_files()
        print('using {} unique words with embedding size of {} '.format(
            self.embedding_np.shape[0],
            self.embedding_np.shape[1]))
