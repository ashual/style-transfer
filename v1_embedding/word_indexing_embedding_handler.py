from v1_embedding.embedding_handler import EmbeddingHandler
import collections
import zipfile
import tensorflow as tf
import json
import numpy as np


class WordIndexingEmbeddingHandler(EmbeddingHandler):
    def __init__(self, filename, embedding_size, n=1, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self)
        vocab = self.build_dataset(self.read_data(filename), n, truncate_by_cutoff)
        # init mappings
        self.vocabulary_to_internals(vocab)
        # init embeddings
        self.embedding_np = np.random.rand(len(vocab), embedding_size)

    def read_data(self, filename):
        """Extract the first file enclosed in a zip file as a list of words."""
        if filename[-3:] == 'zip':
            with zipfile.ZipFile(filename) as f:
                data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        elif 'reviews' in filename:
            jdec = json.JSONDecoder()
            data = []
            with open(filename) as f:
                for line in f:
                    sentence = jdec.decode(line)['text'][:-1].lower()
                    data.extend(sentence.split())
            f.close()
        return data

    def build_dataset(self, words, n, truncate_by_cutoff):
        """Process raw inputs into a dataset."""
        vocab = [self.start_of_sentence_token, self.end_of_sentence_token, self.unknown_token, self.pad_token]
        if truncate_by_cutoff:
            vocab += [w for w, c in collections.Counter(words).most_common() if c >= n]
        else:
            vocab += [w for w, _ in collections.Counter(words).most_common(n - 1)]
        return vocab
