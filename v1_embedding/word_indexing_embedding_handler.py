from v1_embedding.embedding_handler import EmbeddingHandler
import collections
import numpy as np
from nltk import word_tokenize


class WordIndexingEmbeddingHandler(EmbeddingHandler):
    def __init__(self, dataset, embedding_size, n=1, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self)
        vocab = self.build_dataset(WordIndexingEmbeddingHandler.read_data(dataset), n, truncate_by_cutoff)
        # init mappings
        self.vocabulary_to_internals(vocab)
        # init embeddings
        self.embedding_np = np.random.rand(len(vocab), embedding_size)

    @staticmethod
    def read_data(dataset):
        data = []
        for sentence in dataset.get_content():
            data.extend(word_tokenize(sentence))
        return data

    def build_dataset(self, words, n, truncate_by_cutoff):
        """Process raw inputs into a dataset."""
        vocab = [self.start_of_sentence_token, self.end_of_sentence_token, self.unknown_token, self.pad_token]
        if truncate_by_cutoff:
            vocab += [w for w, c in collections.Counter(words).most_common() if c >= n]
        else:
            vocab += [w for w, _ in collections.Counter(words).most_common(n - 1)]
        return vocab
