from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np
from os import getcwd
from os.path import join


class GloveEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, pretrained_glove_file=None, dataset=None):
        EmbeddingHandler.__init__(self, save_dir)
        if not self.initialized_from_cache:
            if pretrained_glove_file is None:
                self.pretrained_glove_file = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")
            else:
                self.pretrained_glove_file = pretrained_glove_file
            print('counting words in glove')
            word_set = dataset.get_word_dictionary()
            print('word_set', len(word_set))
            vocab, self.embedding_np = self.load_from_files(word_set)
            self.vocabulary_to_internals(vocab)
            self.save_files()
            print('used glove for embedding')

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
            vocab += [token]
            extreme_value = np.max(np.abs(embedding))
            new_token_vector = np.array([extreme_value] * embedding.shape[1]).reshape((1, -1))
            new_token_vector *= np.random.choice([1, -1], size=new_token_vector.shape)
            embedding = np.concatenate((embedding, new_token_vector), axis=0)
            return vocab, embedding

        vocab, embd = load_glove(self.pretrained_glove_file, word_dict)
        embedding = np.asarray(embd, dtype=np.float32)
        if self.end_of_sentence_token not in vocab:
            vocab, embedding = set_new_token(self.end_of_sentence_token, vocab, embedding)
        if self.unknown_token not in vocab:
            vocab, embedding = set_new_token(self.unknown_token, vocab, embedding)
        if self.pad_token not in vocab:
            vocab, embedding = set_new_token(self.pad_token, vocab, embedding)
        return vocab, embedding

