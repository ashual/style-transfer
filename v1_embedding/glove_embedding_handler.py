from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np
from os import getcwd
from os.path import join


class GloveEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, pretrained_glove_file=None, force_vocab=False):
        EmbeddingHandler.__init__(self, save_dir)
        if not self.initialized_from_cache:
            if pretrained_glove_file is None:
                self.pretrained_glove_file = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")
            else:
                self.pretrained_glove_file = pretrained_glove_file
            vocab, self.embedding_np = self.load_from_files()
            self.vocabulary_to_internals(vocab)
            self.save_files()
            print('used glove for embedding')

    def load_from_files(self):
        def load_glove(filename):
            vocab = []
            embd = []
            file = open(filename, 'r') #open(filename, 'r', encoding="utf8")
            for line in file.readlines():
                row = line.strip().split(' ')
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

        vocab, embd = load_glove(self.pretrained_glove_file)
        embedding = np.asarray(embd, dtype=np.float32)
        if self.start_of_sentence_token not in vocab:
            vocab, embedding = set_new_token(self.start_of_sentence_token, vocab, embedding)
        if self.end_of_sentence_token not in vocab:
            vocab, embedding = set_new_token(self.end_of_sentence_token, vocab, embedding)
        if self.unknown_token not in vocab:
            vocab, embedding = set_new_token(self.unknown_token, vocab, embedding)
        if self.pad_token not in vocab:
            vocab, embedding = set_new_token(self.pad_token, vocab, embedding)
        return vocab, embedding

