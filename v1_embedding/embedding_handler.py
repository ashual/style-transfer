import os
import pickle
import numpy as np
import collections
from nltk import word_tokenize

class EmbeddingHandler:
    def __init__(self, save_directory):
        self.pad_token = 'PAD'

        self.end_of_sentence_token = 'END'
        self.unknown_token = 'UNK'

        self.word_to_index = None
        self.index_to_word = None
        self.embedding_np = None

        self.save_directory = save_directory
        self.initialized_from_cache = self.load_files()

    def load_files(self):
        word_to_index_path, index_to_word_path, embedding_np_path = self.get_cache_file_names()
        if os.path.exists(word_to_index_path) and \
                os.path.exists(index_to_word_path) and \
                os.path.exists(embedding_np_path):
            try:
                self.word_to_index = pickle.load(open(word_to_index_path, "rb"))
                self.index_to_word = pickle.load(open(index_to_word_path, "rb"))
                self.embedding_np = np.load(embedding_np_path)
                print('initialized embedding from cache')
                return True
            except:
                self.word_to_index = None
                self.index_to_word = None
                self.embedding_np = None
        return False

    def save_files(self):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        word_to_index_path, index_to_word_path, embedding_np_path = self.get_cache_file_names()
        try:
            pickle.dump(self.word_to_index, open(word_to_index_path, 'wb'))
            pickle.dump(self.index_to_word, open(index_to_word_path, 'wb'))
            np.save(embedding_np_path, self.embedding_np)
            return True
        except Exception as e:
            print(str(e))
            return False

    def get_cache_file_names(self):
        word_to_index_path = os.path.join(self.save_directory, 'w2i.p')
        index_to_word_path = os.path.join(self.save_directory, 'i2w.p')
        embedding_np_path = os.path.join(self.save_directory, 'embedding.npy')
        return word_to_index_path, index_to_word_path, embedding_np_path

    def vocabulary_to_internals(self, vocabulary):
        self.index_to_word = {i: w for i, w in enumerate(vocabulary)}
        self.word_to_index = {self.index_to_word[i]: i for i in self.index_to_word}

    def get_word_to_index(self, sentences):
        return [
            [
                self.word_to_index[w] if w in self.word_to_index else self.word_to_index[self.unknown_token] for w in s
            ]
            for s in sentences
        ]

    def get_index_to_word(self, sentences_with_indices):
        return [[self.index_to_word[i] for i in s if i < self.get_vocabulary_length()] for s in sentences_with_indices]

    def get_vocabulary(self):
        return list(self.word_to_index.keys())

    def get_vocabulary_length(self):
        return len(self.word_to_index)

    def get_embedding_size(self):
        return self.embedding_np.shape[1]

    def get_embedding_array(self):
        return self.embedding_np

    @staticmethod
    def read_data(datasets):
        data = []
        for dataset in datasets:
            for sentence in dataset.get_content():
                data.extend(word_tokenize(sentence))
        return data

    def build_dataset(self, datasets, n, truncate_by_cutoff):
        """Process raw inputs into a dataset."""
        words = EmbeddingHandler.read_data(datasets)
        vocab = [self.end_of_sentence_token, self.unknown_token]
        if truncate_by_cutoff:
            vocab += [w for w, c in collections.Counter(words).most_common() if c >= n]
        else:
            vocab += [w for w, _ in collections.Counter(words).most_common(n - 1)]
        return vocab