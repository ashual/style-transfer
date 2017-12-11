import tensorflow as tf

from regina_files.nn import cnn
from regina_files.vocab import Vocabulary


class Model(object):
    def __init__(self, vocab):
        dim_emb = 100
        filter_sizes = [int(x) for x in [3, 4, 5]]
        n_filters = 128

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.x = tf.placeholder(tf.int32, [None, None],  # batch_size * max_len
                                name='x')
        self.y = tf.placeholder(tf.float32, [None], name='y')

        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        x = tf.nn.embedding_lookup(embedding, self.x)
        self.logits = cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')
        self.probs = tf.sigmoid(self.logits)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()


def load_model(sess, model):
    saver_dir = './regina_files/classifier_data/'
    print('Loading classifier model from {}'.format(saver_dir))
    checkpoint_path = tf.train.get_checkpoint_state(saver_dir)
    if checkpoint_path is not None:
        model.saver.restore(sess, checkpoint_path.model_checkpoint_path)
    else:
        raise Exception('no classifier model')
    return model


def evaluate(sess, batch_size, vocab, model, x, y):
    probs = []
    batches = get_batches(x, y, vocab.word2id, batch_size)
    for batch in batches:
        p = sess.run(model.probs, feed_dict={model.x: batch['x'], model.dropout: 1})
        probs += p.tolist()
    y_hat = [p > 0.5 for p in probs]
    same = [p == q for p, q in zip(y, y_hat)]
    return 100.0 * sum(same) / len(y), probs


def get_batches(x, y, word2id, batch_size, min_len=5):
    pad = word2id['<pad>']
    unk = word2id['<unk>']

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))

        _x = []
        max_len = max([len(sent) for sent in x[s:t]])
        max_len = max(max_len, min_len)
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            padding = [pad] * (max_len - len(sent))
            _x.append(padding + sent_id)

        batches.append({'x': _x, 'y': y[s:t]})
        s = t

    return batches


def prepare(sentences):
    x = [' '.join(s) for s in sentences]
    y = [1] * len(x)
    z = sorted(zip(x, y), key=lambda i: len(i[0]))
    return list(zip(*z))


class Classifier:
    def __init__(self, batch_size=100):
        self.vocab = Vocabulary('./regina_files/classifier_data/yelp.vocab')
        print('vocabulary size', self.vocab.size)
        self.batch_size = batch_size
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        c_language = tf.Graph()
        with c_language.as_default():
            model = Model(self.vocab)
            self.sess = tf.Session(config=config, graph=c_language)
            self.model = load_model(self.sess, model)

    def test(self, sentences):
        test_x, test_y = prepare(sentences)
        acc, _ = evaluate(self.sess, self.batch_size, self.vocab, self.model, test_x, test_y)
        print('dev accuracy %.2f' % acc)
        return acc
