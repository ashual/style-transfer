import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingEncoder(BaseModel):
    """
        use self.inputs and self.domain_identifier in order to call self.encoded_vector.
        can also take the encoder model using self.multilayer_encoder
    """
    def __init__(self, embedding_size, hidden_states, context_vector_size):
        # placeholders:
        # domain identifier
        self.domain_identifier = tf.placeholder(tf.int32, shape=())
        domain_identifier = self.print_tensor_with_shape(self.domain_identifier, "domain_identifier")
        # the input sequence s.t (batch, time, embedding)
        self.inputs = tf.placeholder(tf.float32, shape=(None, None, embedding_size))
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")

        # important sizes
        batch_size = tf.shape(inputs)[0]
        sentence_length = tf.shape(inputs)[1]

        # create the input: (batch, time, embedding;domain)
        domain_identifier_tiled = tf.tile(tf.expand_dims(tf.exp(domain_identifier, 0), 0),
                                          [batch_size, sentence_length, 1])
        encoder_inputs = tf.concat((inputs, domain_identifier_tiled), axis=2)

        # encoder - model
        with tf.variable_scope('encoder'):
            encoder_cells = []
            for hidden_size in hidden_states:
                encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(context_vector_size, state_is_tuple=True))
            self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)

        # run the encoder
        with tf.variable_scope('encoder_run'):
            # define the initial state as empty
            initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
            # run the model
            rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_encoder, encoder_inputs,
                                               initial_state=initial_state,
                                               time_major=False)
            encoded_vector = self.print_tensor_with_shape(rnn_outputs[:, -1, :], "encoded")

        # the computational step for running the model
        self.encoded_vector = encoded_vector
