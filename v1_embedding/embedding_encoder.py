import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingEncoder(BaseModel):
    def __init__(self, hidden_states, context_vector_size, namespace_prefix=None):
        BaseModel.__init__(self)
        if namespace_prefix is not None and not namespace_prefix[-1] == '_':
            namespace_prefix += '_'
        self.namespace_prefix = namespace_prefix

        # placeholders:
        # domain identifier
        self.domain_identifier = tf.placeholder(tf.int32, shape=())

        # encoder - model
        with tf.variable_scope('{}encoder'.format(self.namespace_prefix)):
            encoder_cells = []
            for hidden_size in hidden_states:
                encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(context_vector_size, state_is_tuple=True))
            self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)

    def encode_inputs_to_vector(self, inputs):
        with tf.variable_scope('{}encoder_preprocessing'.format(self.namespace_prefix)):
            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, "inputs")

            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # create the input: (batch, time, embedding; domain)
            domain_identifier = self.print_tensor_with_shape(self.domain_identifier, "domain_identifier")
            domain_identifier_tiled = tf.tile(tf.expand_dims(tf.exp(domain_identifier, 0), 0),
                                              [batch_size, sentence_length, 1])
            encoder_inputs = tf.concat((inputs, domain_identifier_tiled), axis=2)

        # run the encoder
        with tf.variable_scope('{}encoder_run'.format(self.namespace_prefix)):
            # define the initial state as empty
            initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
            # run the model
            rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_encoder, encoder_inputs,
                                               initial_state=initial_state,
                                               time_major=False)
            return self.print_tensor_with_shape(rnn_outputs[:, -1, :], "encoded")
