from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np
from os import getcwd
from os.path import join


class GloveEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, datasets, pretrained_embedding_file, embedding_size=200, n=2,
                 truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self, save_dir)
        if not self.initialized_from_cache:
            if pretrained_embedding_file is None:
                self.pretrained_embedding_file = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")
            else:
                if embedding_size == 128:
                    self.pretrained_embedding_file = pretrained_embedding_file
                else:
                     raise Exception('We should train different embedding file')
            print('counting words in pretrained embedding file')
            word_set = self.build_dataset(datasets, n, truncate_by_cutoff)
            print('word_set', len(word_set))
            vocab, self.embedding_np = self.load_from_files(word_set)
            self.vocabulary_to_internals(vocab)
            self.save_files()
        print('using {} unique words with embedding size of {} '.format(
            self.embedding_np.shape[0],
            self.embedding_np.shape[1]))

    def load_from_files(self, word_dict):
        def load_glove(filename, word_dictionary):
            vocab = []
            embd = []
            file = open(filename, 'r')
            for line in file.readlines():
                row = line.strip().split(' ')
                if row[0] in word_dictionary:
                    vocab.append(row[0])
                    embd.append(row[1:])
            print('Loaded GloVe!')
            file.close()
            return vocab, embd

        def set_new_token(token, vocab, embedding):
            raise Exception("set_new_token called")
            # vocab += [token]
            # extreme_value = np.max(np.abs(embedding))
            # new_token_vector = np.array([extreme_value] * embedding.shape[1]).reshape((1, -1))
            # new_token_vector *= np.random.choice([1, -1], size=new_token_vector.shape)
            # embedding = np.concatenate((embedding, new_token_vector), axis=0)
            # return vocab, embedding

        vocab, embd = load_glove(self.pretrained_embedding_file, word_dict)
        embedding = np.asarray(embd, dtype=np.float32)
        if self.end_of_sentence_token not in vocab:
            vocab, embedding = set_new_token(self.end_of_sentence_token, vocab, embedding)
        if self.unknown_token not in vocab:
            vocab, embedding = set_new_token(self.unknown_token, vocab, embedding)
        return vocab, embedding

