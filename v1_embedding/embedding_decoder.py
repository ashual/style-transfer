import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingDecoder(BaseModel):
    """
    there are several exposed methods
    1. use domain_identifier, encoded_vector, inputs and initial_decoder_state to do a prediction.
    returns decoded_vector and decoded_last_state
    call this method iteratively for inputs of size 1 to do sequential prediction (not teacher helping mode)

    2. use decoder_zero_state to get the zero state of the decoder (can be used to execute the above method)
    """
    def __init__(self, embedding_size, hidden_states, context_vector_size):
        # placeholders:
        # domain identifier
        self.domain_identifier = tf.placeholder(tf.int32, shape=())

        # decoder - model
        with tf.variable_scope('decoder'):
            decoder_cells = []
            for hidden_size in hidden_states:
                decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True))
            self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)

    def get_zero_state(self, batch_size):
        with tf.variable_scope('decoder_get_zero_state'):
            return self.multilayer_decoder.zero_state(batch_size, tf.float32)

    def decode_vector_to_sequence(self, encoded_vector, initial_decoder_state, inputs):
        with tf.variable_scope('decoder_preprocessing'):
            # encoded vector (batch, context)
            encoded_vector = self.print_tensor_with_shape(encoded_vector, "encoded_vector")
            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, "inputs")

            domain_identifier = self.print_tensor_with_shape(self.domain_identifier, "domain_identifier")
            initial_decoder_state = self.print_tensor_with_shape(initial_decoder_state, "initial_decoder_state")

            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

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
            decoded_vector = self.print_tensor_with_shape(decoded_vector, "decoded_vector")
            decoder_last_state = self.print_tensor_with_shape(decoder_last_state, "decoder_last_state")

            return decoded_vector, decoder_last_state

