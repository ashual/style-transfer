import tensorflow as tf
from input_utils import InputPipeline, EmbeddingHandler


class EncoderDecoderReconstruction:
    def __init__(self, vocabulary_size, embedding_size, hidden_vector_size, number_of_layers, learning_rate = 0.01):
        self.should_print = tf.placeholder_with_default(False, shape=())
        # all inputs should have a start of sentence and end of sentence tokens
        self.inputs = tf.placeholder(tf.int32, (None, None))  # (batch, time, in)
        inputs = tf.identity(self.inputs)
        # inputs = self.print_tensor_with_shape(inputs, "inputs")

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

# create input pipeline
embedding_handler = EmbeddingHandler(pretrained_glove_file=r"C:\temp\data\style\glove.6B\glove.6B.50d.txt",
                                     force_vocab=False, start_of_sentence_token='START', unknown_token='UNK')
input_stream = InputPipeline(text_file=r"C:\Users\user\Dropbox\projects\StyleTransfer\yoda\english_yoda.text",
                             embedding_handler=embedding_handler)

# model
model = EncoderDecoderReconstruction(embedding_handler.vocab_len, embedding_handler.embedding_size,
                                     hidden_vector_size=100, number_of_layers=2)
session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(10):
    epoch_error = 0
    data_iter = input_stream.batch_iterator(shuffle=True, maximal_batch=2)
    for i, (batch, indexed_batch) in enumerate(data_iter):
        epoch_error += session.run([model.loss, model.train], {
            model.inputs: indexed_batch,
        })[0]
    epoch_error /= (i+1)
    print("Epoch %d, train error: %.2f, %%" % (epoch, epoch_error))

    # TODO: add validation error + hidden vector of the same sentence
    # validation_error = 0
    # data_iter = test_pipe.batch_iterator(1, shuffle=True)
    # for i, (x, y) in enumerate(data_iter):
    #     validation_error += session.run(lstm_model.accuracy, {
    #         lstm_model.inputs:  x,
    #         lstm_model.outputs: y,
    #     })
    # validation_error /= (i+1)
    # print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, validation_error * 100.0))





