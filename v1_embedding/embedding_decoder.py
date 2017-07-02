import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingDecoder(BaseModel):
    def __init__(self, embedding_size, hidden_states):
        # placeholders:
        # domain identifier
        self.domain_identifier = tf.placeholder(tf.int32, shape=())
        domain_identifier = self.print_tensor_with_shape(self.domain_identifier, "domain_identifier")
        # the input sequence s.t (batch, time, embedding)
        self.inputs = tf.placeholder(tf.float32, shape=(None, None, embedding_size))
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")
        # encoded vector (batch, embedding)
        self.encoded_vector = tf.placeholder(tf.float32, shape=(None, embedding_size))
        encoded_vector = self.print_tensor_with_shape(self.encoded_vector, "encoded_vector")

        # important sizes
        batch_size = tf.shape(inputs)[0]
        sentence_length = tf.shape(inputs)[1]

        # decoder - model
        with tf.variable_scope('decoder'):
            decoder_cells = []
            for hidden_size in hidden_states:
                decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True))
            self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)

        self.decoder_initial_state = self.multilayer_decoder.zero_state(batch_size, tf.float32)

        # initial decoder state
        self.initial_decoder_state = tf.placeholder(tf.float32, shape=tf.shape(self.decoder_initial_state))
        initial_decoder_state = self.print_tensor_with_shape(self.initial_decoder_state, "initial_decoder_state")

        # the decoder input need to append encoded vector to embedding of each input
        decoder_inputs = tf.expand_dims(encoded_vector, 1)
        decoder_inputs = tf.tile(decoder_inputs, [1, sentence_length, 1])
        decoder_inputs = tf.concat((inputs, decoder_inputs), axis=2)
        domain_identifier_tiled = tf.tile(tf.expand_dims(tf.exp(domain_identifier, 0), 0),
                                          [batch_size, sentence_length, 1])
        decoder_inputs = tf.concat((decoder_inputs, domain_identifier_tiled), axis=2)
        decoder_inputs = self.print_tensor_with_shape(decoder_inputs, "decoder_inputs")

        with tf.variable_scope('decoder_run'):
            decoded_vector, decoder_last_state = tf.nn.dynamic_rnn(self.multilayer_decoder, decoder_inputs,
                                                                   initial_state=initial_decoder_state,
                                                                   time_major=False)
            self.decoded_vector = self.print_tensor_with_shape(decoded_vector, "decoded_vector")
            self.decoder_last_state = self.print_tensor_with_shape(decoder_last_state, "decoder_last_state")

