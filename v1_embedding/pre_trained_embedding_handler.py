from v1_embedding.embedding_handler import EmbeddingHandler
import numpy as np
from os import getcwd
from os.path import join


class PreTrainedEmbeddingHandler(EmbeddingHandler):
    def __init__(self, save_dir, datasets, type, embedding_size=200, n=2, truncate_by_cutoff=True):
        EmbeddingHandler.__init__(self, save_dir)
        if not self.initialized_from_cache:
            if embedding_size == 100 or embedding_size == 200:
                if type == 'BIBLE':
                    self.pretrained_embedding_file = join(getcwd(), 'data', 'embeddings-bibleASVandKJV-3114-{}-2.txt'.format(embedding_size))
                else:
                    self.pretrained_embedding_file = \
                        join(getcwd(), 'data', "embeddings-53708-{}-2.txt".format(embedding_size))
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
            print('Loaded {}!'.format(filename))
            file.close()
            return vocab, embd

        vocab, embd = load_glove(self.pretrained_embedding_file, word_dict)
        embedding = np.asarray(embd, dtype=np.float32)
        if self.end_of_sentence_token not in vocab or self.unknown_token not in vocab:
            raise Exception("end or unknown token does not exist")
        return vocab, embedding
