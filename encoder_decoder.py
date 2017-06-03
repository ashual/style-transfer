import tensorflow as tf
from input_utils import InputPipeline, EmbeddingHandler
import copy
from os import getcwd
from os.path import join
import timeit


class EncoderDecoderReconstruction:
    def __init__(self, vocabulary_size, embedding_size, hidden_states, learning_rate=0.00001,
                 cross_entropy_weight=0.001, reconstruction_weight=1.0):

        # placeholders:
        self.should_print = tf.placeholder_with_default(False, shape=())
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])
        # all inputs should have a <START> token before every sentence
        self.inputs = tf.placeholder(tf.int32, (None, None))  # (batch, time)=> index of word

        def __graph__():
            inputs = self.print_tensor_with_shape(self.inputs, "inputs")

            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # set the inputs for the encoder and the decoder
            encoder_inputs = inputs[:, 1:]  # skip the <START> token
            decoder_inputs = inputs[:, :-1]  # all but the last word

            # embedding
            w = tf.Variable(tf.random_normal(shape=[vocabulary_size, embedding_size]),
                            trainable=False, name="word_vectors")

            # you can call assign embedding op to init the embedding
            self.assign_embedding = tf.assign(w, self.embedding_placeholder)
            self.w = w

            # holds all the relevant sizes of the output sizes for the RNNs
            layers_sizes = copy.deepcopy(hidden_states)
            layers_sizes.insert(0, embedding_size)

            # encoder
            with tf.variable_scope('encoder'):
                encoder_cells = []
                for hidden_size in layers_sizes[1:]:
                    encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))

                self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)

            # the last output of the encoder provides the hidden vector
            embedding_encoder, self.encoded_vector = self.run_encoder(encoder_inputs, batch_size)
            encoded_vector = self.print_tensor_with_shape(self.encoded_vector, "encoded_vector")

            # decoder
            with tf.variable_scope('decoder'):
                decoder_cells = []
                layers_sizes.reverse()  # the layers are mirror image of the encoder
                for hidden_size in layers_sizes[1:]:
                    decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
                self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)

            decoder_initial_state = self.multilayer_decoder.zero_state(batch_size, tf.float32)

            self.decoded_vector = self.run_decoder(decoder_inputs, encoded_vector, decoder_initial_state)[0]
            decoded_vector = self.print_tensor_with_shape(self.decoded_vector, "decoded_vector")

            # linear layer to project to vocabulary
            decoder_reshaped = tf.reshape(decoded_vector, (batch_size * (sentence_length-1), embedding_size))
            self.linear_w = tf.Variable(tf.random_normal(shape=(embedding_size, vocabulary_size)), dtype=tf.float32)
            self.linear_b = tf.Variable(tf.random_normal(shape=(vocabulary_size,)), dtype=tf.float32)
            vocab_logits = self.run_linear_layer(decoder_reshaped)

            # compute loss for the embedding
            loss_embedding = tf.reduce_mean(tf.squared_difference(embedding_encoder, decoded_vector))

            # and compute loss for the predicted softmax
            inputs_reshaped = tf.reshape(encoder_inputs, (batch_size * (sentence_length - 1),))
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs_reshaped, logits=vocab_logits))

            self.loss = reconstruction_weight * loss_embedding + cross_entropy_weight * cross_entropy_loss

            # train step
            self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        __graph__()

    def run_encoder(self, encoder_inputs, batch_size):
        embedding_encoder = tf.nn.embedding_lookup(self.w, encoder_inputs)
        embedding_encoder = self.print_tensor_with_shape(embedding_encoder, "embedding_encoder")

        with tf.variable_scope('encoder_run'):
            initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_encoder, embedding_encoder,
                                               initial_state=initial_state,
                                               time_major=False)
        encoded_vector = rnn_outputs[:, -1, :]
        return embedding_encoder, encoded_vector

    def run_decoder(self, decoder_inputs, encoded_vector, initial_state):
        embedding_decoder = tf.nn.embedding_lookup(self.w, decoder_inputs)
        embedding_decoder = self.print_tensor_with_shape(embedding_decoder, "embedding_decoder")

        # for the decoder input need to append encoded vector to embedding of each input
        decoder_inputs_with_embeddings = tf.expand_dims(encoded_vector, 1)
        decoder_inputs_with_embeddings = tf.tile(decoder_inputs_with_embeddings, [1, tf.shape(embedding_decoder)[1], 1])
        decoder_inputs_with_embeddings = tf.concat((embedding_decoder, decoder_inputs_with_embeddings), axis=2)
        decoder_inputs_with_embeddings = self.print_tensor_with_shape(decoder_inputs_with_embeddings,
                                                                      "decoder_inputs_with_embeddings")

        with tf.variable_scope('decoder_run'):
            decoded_vector, last_state = tf.nn.dynamic_rnn(self.multilayer_decoder, decoder_inputs_with_embeddings,
                                                           initial_state=initial_state, time_major=False)
        return decoded_vector, last_state

    def run_linear_layer(self, linear_input):
        return tf.matmul(linear_input, self.linear_w) + self.linear_b

    def do_sequence_prediction(self, start_index, stop_index, max_prediction_length):
        # we take the inputs as usual
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")
        # make sure only one sentence
        tf.assert_equal(tf.shape(inputs)[0], 1)

        should_stop_due_to_length = tf.not_equal(max_prediction_length, -1)

        # get the encoded vector
        encoder_inputs = inputs[:, 1:]  # skip the <START> token
        encoded_vector = self.run_encoder(encoder_inputs, 1)[1]

        decoder_input = tf.Variable(tf.constant(start_index, shape=(1, 1)), trainable=False)
        decoder_state = self.multilayer_decoder.zero_state(1, tf.float32)
        iteration_counter = tf.Variable(trainable=False)
        result = tf.zeros(shape=(0, ), dtype=tf.int32)

        def while_cond(decoder_input, decoder_state, iteration_counter, result):
            is_max_iteration =  tf.logical_and(should_stop_due_to_length,
                                               tf.equal(iteration_counter, max_prediction_length))
            is_stop_token = tf.equal(decoder_input, stop_index)
            return tf.logical_or(is_max_iteration, is_stop_token)

        def while_body(decoder_input, decoder_state, iteration_counter, result):
            iteration_counter += 1
            decoded_vector, decoder_state = self.run_decoder(decoder_input, encoded_vector, decoder_state)
            linear_projection = self.run_linear_layer(decoded_vector)
            decoder_input = tf.argmax(linear_projection)
            result = tf.concat((result, decoder_input), axis=0)
            return [decoder_input, decoder_state, iteration_counter, result]

        output = tf.while_loop(cond=while_cond, body=while_body,
                               loop_vars=[decoder_input, decoder_state, iteration_counter, result],
                               shape_invariants=[
                                   decoder_input.get_shape(),
                                   decoder_state.get_shape(),
                                   iteration_counter.get_shape(),
                                   tf.TensorShape(tf.Dimension(None))
                               ], parallel_iterations=1, back_prop=False)[3]
        return output


    def print_tensor_with_shape(self, tensor, name):
        return tf.cond(self.should_print,
                       lambda: tf.Print(
                           tf.Print(tensor, [tensor], message=name + ":"),
                           [tf.shape(tensor)], message=name + " shape:"),
                       lambda: tf.identity(tensor))


glove_file = join(getcwd(), "data", "glove.6B", "glove.6B.50d.txt")
yoda_file = join(getcwd(), "yoda", "english_yoda.text")

# create input pipeline
embedding_handler = EmbeddingHandler(pretrained_glove_file=glove_file,
                                     force_vocab=False, start_of_sentence_token='START', unknown_token='UNK')
input_stream = InputPipeline(text_file=yoda_file, embedding_handler=embedding_handler, limit_sentences=10)

model = EncoderDecoderReconstruction(embedding_handler.vocab_len, embedding_handler.embedding_size,
                                     hidden_states=[10, 5], cross_entropy_weight=1.0, reconstruction_weight=1.0,
                                     learning_rate=0.001)
session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

# init embedding
_ = session.run(model.assign_embedding, {model.embedding_placeholder: embedding_handler.embedding_np})

for epoch in range(100):
    start_time = timeit.default_timer()
    epoch_loss = 0
    num_sentences = 0
    data_iter = input_stream.batch_iterator(shuffle=True, maximal_batch=2)
    for batch, indexed_batch in data_iter:
        num_sentences += len(batch)
        epoch_loss += session.run([model.loss, model.train], {
            model.inputs: indexed_batch,
        })[0]
    epoch_loss /= num_sentences
    elapsed = timeit.default_timer() - start_time
    print("Epoch %d, elapsed time: %.2f train error: %.2f" % (epoch, elapsed, epoch_loss))

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





