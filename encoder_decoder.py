import tensorflow as tf
from input_utils import InputPipeline, EmbeddingHandler


class EncoderDecoderReconstruction:
    def __init__(self, vocabulary_size, embedding_size, hidden_states, learning_rate=0.00001, words_loss_constant=0.001):
        self.should_print = tf.placeholder_with_default(False, shape=())
        # all inputs should have a start of sentence and end of sentence tokens
        self.inputs = tf.placeholder(tf.int32, (None, None))  # (batch, time, in)
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")

        # embedding
        # may change to https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence
        w = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embedding_size]),
                        trainable=False, name="WordVectors")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])
        w = w.assign(self.embedding_placeholder)
        self.embedding = tf.nn.embedding_lookup(w, inputs)
        embedding = self.print_tensor_with_shape(self.embedding, "embedding")

        # holds all the relevant sizes of the output sizes for the RNNs
        all_state_sizes = hidden_states.copy()
        all_state_sizes.insert(0, embedding_size)

        # important sized
        batch_size = tf.shape(inputs)[0]
        sentence_length = tf.shape(inputs)[1]

        # encoder
        with tf.variable_scope('encoder'):
            encoder_cells = []
            for hidden_size in all_state_sizes[1:]:
                encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))

            multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            initial_state = multilayer_encoder.zero_state(batch_size, tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(multilayer_encoder, embedding,
                                               initial_state=initial_state,
                                               time_major=False)
        self.encoded_vector = rnn_outputs[:, -1, :]
        encoded_vector = self.print_tensor_with_shape(self.encoded_vector, "encoded_vector")

        # decoder input - append encoded vector to embedding of each input
        decoder_inputs = tf.expand_dims(encoded_vector, 1)
        timesteps = tf.shape(rnn_outputs)[1]
        decoder_inputs = tf.tile(decoder_inputs, [1, timesteps, 1])
        decoder_inputs = tf.concat((embedding, decoder_inputs), axis=2)
        decoder_inputs = self.print_tensor_with_shape(decoder_inputs, "decoder_inputs")

        # decoder
        with tf.variable_scope('decoder'):
            decoder_cells = []
            all_state_sizes.reverse()
            for hidden_size in all_state_sizes[1:]:
                decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            initial_state = multilayer_decoder.zero_state(batch_size, tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(multilayer_decoder, decoder_inputs,
                                               initial_state=initial_state,
                                               time_major=False)
        self.decoded_vector = rnn_outputs
        decoded_vector = self.print_tensor_with_shape(self.decoded_vector, "decoded_vector")

        # compute loss
        loss_embedding = tf.reduce_mean(tf.squared_difference(embedding, decoded_vector))
        self.loss = loss_embedding

        if words_loss_constant > 0.0:
            # linear layer
            decoder_reshaped = tf.reshape(decoded_vector, (batch_size * sentence_length, embedding_size))
            vocab_logits = tf.contrib.layers.linear(decoder_reshaped, vocabulary_size)
            logits_reshaped = tf.reshape(vocab_logits, (batch_size, sentence_length, vocabulary_size))
            logits_argmax = tf.reduce_max(logits_reshaped, axis=2)

            loss_words = words_loss_constant * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits_argmax))
            self.loss += loss_words

        # train step
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def print_tensor_with_shape(self, tensor, name):
        return tf.cond(self.should_print,
                       lambda: tf.Print(
                           tf.Print(tensor, [tensor], message=name + ":"),
                           [tf.shape(tensor)], message=name + " shape:"),
                       lambda: tf.identity(tensor))

# for first time users do the following in shell:
# import nltk
# nltk.download()

# create input pipeline
embedding_handler = EmbeddingHandler(pretrained_glove_file=r"C:\temp\data\style\glove.6B\glove.6B.50d.txt",
                                     force_vocab=False, start_of_sentence_token='START', unknown_token='UNK')
input_stream = InputPipeline(text_file=r"C:\Users\user\Dropbox\projects\StyleTransfer\yoda\english_yoda.text",
                             embedding_handler=embedding_handler)

model = EncoderDecoderReconstruction(embedding_handler.vocab_len, embedding_handler.embedding_size,
                                     hidden_states=[10, 5], words_loss_constant=0.0, learning_rate=0.001)
session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(100):
    epoch_error = 0
    data_iter = input_stream.batch_iterator(shuffle=True, maximal_batch=2)
    for i, (batch, indexed_batch) in enumerate(data_iter):
        epoch_error += session.run([model.loss, model.train], {
            model.inputs: indexed_batch,
            model.embedding_placeholder: embedding_handler.embedding_np
        })[0]
    epoch_error /= (i+1)
    print("Epoch %d, train error: %.2f" % (epoch, epoch_error))

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





