import tensorflow as tf
import numpy as np
import pickle

PRETRAINED_GLOVE_FILE = r"./data/glove.6B/glove.6B.50d.txt"


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


def get_files(filename):
    vocab_filename = filename + '.vocab'
    np_embedding_file = filename + '.npy'
    embedding_file = filename + '.ckpt'
    return vocab_filename, np_embedding_file, embedding_file


def set_new_token(token, vocab, embedding):
    vocab += [token]
    extreme_value = np.max(np.abs(embedding))
    start_token_vector = np.array([extreme_value] * embedding.shape[1]).reshape((1, -1))
    start_token_vector *= np.random.choice([1,-1], size = start_token_vector.shape)
    embedding = np.concatenate((embedding, start_token_vector), axis=0)
    return vocab, embedding


def save_embedding(filename, start_of_sentence_token=None, unknown_token=None):
    vocab, embd = load_GloVe(filename)
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

    vocab_filename, np_embedding_file, embedding_file = get_files(filename)

    # save vocab
    pickle.dump(vocab, open(vocab_filename, 'wb'))
    # with open(vocab_filename, 'wb', encoding="utf8") as vocab_file:
    #     vocab_file.writelines(vocab)
    # save np embedding
    np.save(np_embedding_file, embedding)
    # save tf embedding
    saver = tf.train.Saver({"WordVectors": w})
    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
        saver.save(sess, embedding_file)
        print("Model saved in file: %s" % embedding_file)


def load_from_files(filename):
    vocab_filename, np_embedding_file, embedding_file = get_files(filename)
    return pickle.load(open(vocab_filename, "rb")), np.load(np_embedding_file), embedding_file

if __name__ == "__main__":
    save_embedding(PRETRAINED_GLOVE_FILE, 'START', 'UNK')

# def load_embedding_example(embedding_file):
#     # we only load the embeddings to get the size of W
#     vocab, embd = load_GloVe(filename)
#     with open(r'C:\temp\vocab.txt','w', encoding="utf8") as f:
#         f.writelines([v + '\n' for v in vocab])
#     vocab_size = len(vocab)
#     embedding_dim = len(embd[0])
#
#     W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                     trainable=False, name="WordVectors")
#     run_me = tf.Print(tf.Print(W, [W], message=W.name + ":"), [tf.shape(W)], message=W.name + " shape:")
#
#     saver = tf.train.Saver({"WordVectors": W})
#     with tf.Session() as sess:
#         saver.restore(sess, embedding_file)
#         print("Model restored.")
#         sess.run(run_me, {})

# load_embedding_example(embedding_file)