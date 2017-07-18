import os.path
import pickle

import numpy as np
import tensorflow as tf
from nltk import download as nltk_download

from os import getcwd
from os.path import join

PRETRAINED_GLOVE_FILE = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")


class EmbeddingHandler:
    def __init__(self, pretrained_glove_file, force_vocab=False, start_of_sentence_token=None,
                 end_of_sentence_token=None, unknown_token=None, pad_token=None):
        self.pretrained_glove_file = pretrained_glove_file
        vocab_filename, np_embedding_file, embedding_file = self.get_pretrained_files()
        # check if need to compute local files
        if not os.path.exists(vocab_filename) or force_vocab:
            self.save_embedding(start_of_sentence_token, end_of_sentence_token, unknown_token, pad_token)
        # load relevant data
        self.vocab, self.embedding_np, tf_embedding_filepath = self.load_from_files()
        self.vocab_len = len(self.vocab)
        self.vocabulary_size = len(self.vocab)
        self.embedding_size = self.embedding_np.shape[1]
        self.start_of_sentence_token = self.vocab[-4]
        assert (self.start_of_sentence_token == start_of_sentence_token)
        self.end_of_sentence_token = self.vocab[-3]
        assert (self.end_of_sentence_token == end_of_sentence_token)
        self.unknown_token = self.vocab[-2]
        assert (self.unknown_token == unknown_token)
        self.pad_token = self.vocab[-1]
        assert (self.pad_token == pad_token)
        self.index_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.word_to_index = {self.index_to_word[i]: i for i in self.index_to_word}
        self.start_token_index = self.word_to_index[start_of_sentence_token]
        self.end_token_index = self.word_to_index[end_of_sentence_token]
        self.unknown_token_index = self.word_to_index[unknown_token]
        self.pad_token_index = self.word_to_index[pad_token]

    def get_pretrained_files(self):
        vocab_filename = self.pretrained_glove_file + '.vocab'
        np_embedding_file = self.pretrained_glove_file + '.npy'
        embedding_file = self.pretrained_glove_file + '.ckpt'
        return vocab_filename, np_embedding_file, embedding_file

    def save_embedding(self, start_of_sentence_token, end_of_sentence_token, unknown_token, pad_token):

        def load_GloVe(filename):
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

        vocab, embd = load_GloVe(self.pretrained_glove_file)
        embedding_dim = len(embd[0])
        embedding = np.asarray(embd, dtype=np.float32)
        if start_of_sentence_token is not None:
            vocab, embedding = set_new_token(start_of_sentence_token, vocab, embedding)
        if end_of_sentence_token is not None:
            vocab, embedding = set_new_token(end_of_sentence_token, vocab, embedding)
        if unknown_token is not None:
            vocab, embedding = set_new_token(unknown_token, vocab, embedding)
        if pad_token is not None:
            vocab, embedding = set_new_token(pad_token, vocab, embedding)
        vocab_size = len(vocab)

        w = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                        trainable=False, name="WordVectors")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = w.assign(embedding_placeholder)

        vocab_filename, np_embedding_file, embedding_file = self.get_pretrained_files()

        # save vocab
        pickle.dump(vocab, open(vocab_filename, 'wb'))
        # save np embedding
        np.save(np_embedding_file, embedding)
        # save tf embedding
        saver = tf.train.Saver({"WordVectors": w})
        with tf.Session() as sess:
            sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
            saver.save(sess, embedding_file)
            print("Model saved in file: %s" % embedding_file)

    def load_from_files(self):
        vocab_filename, np_embedding_file, embedding_file = self.get_pretrained_files()
        return pickle.load(open(vocab_filename, "rb")), np.load(np_embedding_file), embedding_file



if __name__ == '__main__':
    # Download NLTK tokenizer
    nltk_download('punkt')
    embedding_handler = EmbeddingHandler(pretrained_glove_file=PRETRAINED_GLOVE_FILE,
                                         force_vocab=False, start_of_sentence_token='START', unknown_token='UNK')
    input_stream = InputPipeline(text_file=r"./yoda/english.text",
                                 embedding_handler=embedding_handler)
    for batch, indexed_batch in input_stream.batch_iterator(shuffle=True, maximal_batch=3):
        print('batch size {}'.format(len(batch)))
        print(batch)
        print(indexed_batch)
        print()
