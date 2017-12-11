from regina_files.nn import *
from regina_files.vocab import Vocabulary
import numpy as np


class Model(object):
    def __init__(self, vocab):
        dim_z = 500
        n_layers = 1

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.inputs = tf.placeholder(tf.int32, [None, None],  # batch_size * max_len
                                     name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None], name='weights')

        embedding = tf.get_variable('embedding', initializer=vocab.embedding.astype(np.float32))
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_z, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        cell = create_cell(dim_z, n_layers, self.dropout)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope='language_model')
        outputs = tf.nn.dropout(outputs, self.dropout)
        outputs = tf.reshape(outputs, [-1, dim_z])
        self.logits = tf.matmul(outputs, proj_W) + proj_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=self.logits)
        loss *= tf.reshape(self.weights, [-1])
        self.tot_loss = tf.reduce_sum(loss)
        self.sent_loss = self.tot_loss / tf.to_float(self.batch_size)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.sent_loss)

        self.saver = tf.train.Saver()


def load_model(sess, model):
    saver_dir = './regina_files/language_data/'
    print('Loading language model from {}'.format(saver_dir))
    checkpoint_path = tf.train.get_checkpoint_state(saver_dir)
    if checkpoint_path is not None:
        model.saver.restore(sess, checkpoint_path.model_checkpoint_path)
    else:
        raise Exception('no language model')
    return model


def get_lm_batches(x, word2id, batch_size):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    x = sorted(x, key=lambda i: len(i))

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))

        go_x, x_eos, weights = [], [], []
        max_len = max([len(sent) for sent in x[s:t]])
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            l = len(sent)
            padding = [pad] * (max_len - l)
            go_x.append([go] + sent_id + padding)
            x_eos.append(sent_id + [eos] + padding)
            weights.append([1.0] * (l + 1) + [0.0] * (max_len - l))

        batches.append({'inputs': go_x, 'targets': x_eos, 'weights': weights, 'size': t - s})
        s = t

    return batches


def evaluate(sess, batch_size, vocab, model, x):
    batches = get_lm_batches(x, vocab.word2id, batch_size)
    tot_loss, n_words = 0, 0

    for batch in batches:
        tot_loss += sess.run(model.tot_loss, feed_dict={model.batch_size: batch['size'], model.inputs: batch['inputs'],
                                                        model.targets: batch['targets'],
                                                        model.weights: batch['weights'], model.dropout: 1})
        n_words += np.sum(batch['weights'])

    return np.exp(tot_loss / n_words)


class LanguageModel:
    def __init__(self, batch_size=100):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.batch_size = batch_size
        self.vocab = Vocabulary('./regina_files/language_data/yelp.vocab', 100)
        print('vocabulary size', self.vocab.size)
        g_language = tf.Graph()
        with g_language.as_default():
            model = Model(self.vocab)
            self.sess = tf.Session(config=config, graph=g_language)
            self.model = load_model(self.sess, model)

    def test(self, sentences):
        ppl = evaluate(self.sess, self.batch_size, self.vocab, self.model, sentences)
        print('dev perplexity %.2f' % ppl)
        return ppl
