import tensorflow as tf
from load_word_embeddings import load_from_files
from nltk import word_tokenize

class EncoderDecoderReconstruction:
    def __init__(self, vocabulary_size, embedding_size, hidden_vector_size, number_of_layers, learning_rate = 0.01):
        self.should_print = tf.placeholder_with_default(False, shape=())
        # all inputs should have a start of sentence and end of sentence tokens
        self.inputs = tf.placeholder(tf.int32, (None, None))  # (batch, time, in)
        inputs = tf.identity(self.inputs)
        inputs = self.print_tensor_with_shape(inputs, "inputs")

        batch_size = tf.shape(inputs)[0]

        w = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embedding_size]),
                        trainable=False, name="WordVectors")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])
        w = w.assign(self.embedding_placeholder)
        self.embedding = tf.nn.embedding_lookup(w, inputs)

        current_input = self.embedding
        hidden_states = EncoderDecoderReconstruction.generate_layers_hidden_sizes(embedding_size, hidden_vector_size,
                                                                                  number_of_layers)
        encoder_hidden_states = []
        for hidden_size in hidden_states[1:]:
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, current_input, initial_state=initial_state,
                                                        time_major=False)
            current_input = rnn_outputs
            encoder_hidden_states.insert(0, rnn_states)
        self.encoded_vector = rnn_states[-1]

        current_input = inputs
        for i, hidden_size in enumerate(encoder_hidden_states.reverse()[1:]):
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
            initial_state = encoder_hidden_states[i][:, -1, :]
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, current_input, initial_state=initial_state,
                                                        time_major=False)
            current_input = rnn_outputs
        self.decoded_vector = rnn_outputs[-1]
        self.loss = tf.reduce_mean(tf.squared_difference(self.embedding, self.decoded_vector))
        self.train = tf.train.AdamOptimizer().minimize(self.loss)



    @staticmethod
    def generate_layers_hidden_sizes(embedding_size, hidden_vector_size, number_of_layers):
        delta = int(float(hidden_vector_size - embedding_size) / (number_of_layers))
        return [l*delta + embedding_size for l in range(number_of_layers+1)]

# for first time users do the following in shell:
# import nltk
# nltk.download()


vocab, embedding_np, embedding_filepath = load_from_files(r"C:\temp\data\style\glove.6B\glove.6B.50d.txt")
with open(r"C:\Users\user\Dropbox\projects\StyleTransfer\yoda\english_yoda.text", "r") as yoda_file:
    yoda_sentences = yoda_file.readlines()

print(len(yoda_sentences))
yoda_tokenized = [word_tokenize(s) for s in yoda_sentences]

vocab = {i:w for i,w in enumerate(vocab)}
reverse_vocab = {vocab[i]:i for i in vocab}

def find_in_vocab(w, reverse_vocab):
    if w in reverse_vocab:
        return reverse_vocab[w]
    return reverse_vocab['UNK']

yoda_indexed = [[find_in_vocab(w, reverse_vocab) for w in s] for s in yoda_tokenized]
print(yoda_sentences[0])
print(yoda_tokenized[0])
print(yoda_indexed[0])


