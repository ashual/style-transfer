import os.path
import tensorflow as tf
import numpy as np
import pickle
import copy
from nltk import download as nltk_download
from nltk import word_tokenize
from load_word_embeddings import PRETRAINED_GLOVE_FILE


class EmbeddingHandler:
    def __init__(self, pretrained_glove_file, force_vocab=False, start_of_sentence_token=None, unknown_token=None):
        self.pretrained_glove_file = pretrained_glove_file
        vocab_filename, np_embedding_file, embedding_file = self.get_pretrained_files()
        # check if need to compute local files
        if not os.path.exists(vocab_filename) or force_vocab:
            self.save_embedding(start_of_sentence_token, unknown_token)
        # load relevant data
        self.vocab, self.embedding_np, tf_embedding_filepath = self.load_from_files()
        self.vocab_len = len(self.vocab)
        self.embedding_size = self.embedding_np.shape[1]
        self.start_of_sentence_token = self.vocab[-2]
        assert (self.start_of_sentence_token == start_of_sentence_token)
        self.unknown_token = self.vocab[-1]
        assert (self.unknown_token == unknown_token)
        self.index_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.word_to_index = {self.index_to_word[i]: i for i in self.index_to_word}

    def get_pretrained_files(self):
        vocab_filename = self.pretrained_glove_file + '.vocab'
        np_embedding_file = self.pretrained_glove_file + '.npy'
        embedding_file = self.pretrained_glove_file + '.ckpt'
        return vocab_filename, np_embedding_file, embedding_file

    def save_embedding(self, start_of_sentence_token, unknown_token):

        def load_GloVe(filename):
            vocab = []
            embd = []
            file = open(filename, 'r')
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
            start_token_vector = np.array([extreme_value] * embedding.shape[1]).reshape((1, -1))
            start_token_vector *= np.random.choice([1, -1], size=start_token_vector.shape)
            embedding = np.concatenate((embedding, start_token_vector), axis=0)
            return vocab, embedding

        vocab, embd = load_GloVe(self.pretrained_glove_file)
        embedding_dim = len(embd[0])
        embedding = np.asarray(embd, dtype=np.float32)
        if start_of_sentence_token is not None:
            vocab, embedding = set_new_token(start_of_sentence_token, vocab, embedding)
        if unknown_token is not None:
            vocab, embedding = set_new_token(unknown_token, vocab, embedding)
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


class InputPipeline:
    def __init__(self, text_file, embedding_handler):
        self.embedding_handler = embedding_handler
        self.text_file = text_file
        with open(text_file, "r") as f:
            self.sentences = f.readlines()

    def batch_iterator(self, shuffle=True, maximal_batch=100):

        def find_in_vocab(w, reverse_vocab):
            if w in reverse_vocab:
                return reverse_vocab[w]
            return reverse_vocab['UNK']

        sentences = copy.deepcopy(self.sentences)
        if shuffle:
            np.random.shuffle(sentences)
        sentences_by_length = {}
        for sentence in sentences:
            indexed_sentence = [find_in_vocab(w, self.embedding_handler.word_to_index) for w in word_tokenize(sentence)]
            len_s = len(indexed_sentence)
            if len_s not in sentences_by_length:
                sentences_by_length[len_s] = []
            sentences_by_length[len_s].append((sentence, indexed_sentence))
        while len(sentences_by_length) > 0:
            len_s = np.random.choice(list(sentences_by_length.keys()), 1)[0]
            current_bucket = sentences_by_length[len_s]
            to_take = np.min((len(current_bucket), maximal_batch))
            # these are the return values
            batch = [x for x, _ in current_bucket[:to_take]]
            indexed_batch = [x for _, x in current_bucket[:to_take]]
            # remove from next iterations
            current_bucket = current_bucket[to_take:]
            if len(current_bucket) == 0:
                sentences_by_length.pop(len_s)
            else:
                sentences_by_length[len_s] = current_bucket
            # return the result
            yield batch, indexed_batch


if __name__ == '__main__':
    # Download NLTK tokenizer
    nltk_download('punkt')
    embedding_handler = EmbeddingHandler(pretrained_glove_file=PRETRAINED_GLOVE_FILE,
                                         force_vocab=False, start_of_sentence_token='START', unknown_token='UNK')
    input_stream = InputPipeline(text_file=r"./yoda/english_yoda.text",
                                 embedding_handler=embedding_handler)
    for batch, indexed_batch in input_stream.batch_iterator(shuffle=True, maximal_batch=3):
        print('batch size {}'.format(len(batch)))
        print(batch)
        print(indexed_batch)
        print()
