#running remarks: embedding_size (100 or 200) and threshold (minimal number of occurences). repeat for 100, 200, 1 and 2.



# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import json
import datetime
import yaml

from nltk import word_tokenize

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


filenames = ["datasets/yelp/negative_reviews.json", "datasets/yelp/positive_reviews.json"]
with open("config/gan.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)
embedding_size = config['embedding']['word_size'] # Dimension of the embedding vector.
threshold = config['embedding']['min_word_occurrences']


def read_data(filenames):
    data = []
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                parsed_line = json.loads(line)
                sentence = parsed_line["text"].lower()
                words = word_tokenize(sentence)
                words += ["END"]
                data.extend(words)
    return data

vocabulary = read_data(filenames)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.

def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    # count.extend(collections.Counter(words).most_common(n_words - 1))
    count += [[w, c] for w, c in collections.Counter(words).most_common() if c >= threshold]
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            # buffer[:] = data[:span]
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# Step 4: Build and train a skip-gram model.

batch_size = 128
skip_window = 2  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

vocabulary_size = len(count)
graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.

def save_embeddings(step, final):
    if final:
        embeddings_source = final_embeddings
    else:
        embeddings_source = normalized_embeddings.eval()
    print(len(embeddings_source[0]) == embedding_size)

    filename = "embeddings-"+str(count[0][1])+"-"+str(embedding_size) + "-" + str(threshold) + "-" + str(datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')) + str(step) + ".txt"
    file = open(filename, "w")
    for i in range(vocabulary_size):
        emb = ""
        for j in range(embedding_size):
            emb += str(embeddings_source[i][j]) + " "
        s = str(count[i][0]) + " " + emb + "\n"
        file.write(s)
    file.close()
    print(len(embeddings_source[0]) == embedding_size)


num_steps = 20000000

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 200000 == 0:
            if step > 0:
                average_loss /= 200000
            # The average loss is an estimate of the loss over the last --- batches.
            print('Average loss at step ', step, ': ', average_loss)
            # save_embeddings(step, False)
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()


save_embeddings(0, True)



    # # Step 6: Visualize the embeddings.
#
#
# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
#     plt.figure(figsize=(18, 18))  # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(label,
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#
#     plt.savefig(filename)
#
#
# try:
#     # pylint: disable=g-import-not-at-top
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#     print('Please install sklearn, matplotlib, and scipy to show embeddings.')
