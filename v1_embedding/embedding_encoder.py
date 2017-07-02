import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingEncoder(BaseModel):
    def __init__(self, embedding_size, hidden_states, should_print=False):
        super(EmbeddingEncoder, self).__init__(should_print)

        # placeholders:
        self.inputs = tf.placeholder(tf.float32, (None, None, embedding_size))  # (batch, time, embedding)
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")

        # important sizes
        batch_size = tf.shape(inputs)[0]

        # encoder - model
        with tf.variable_scope('encoder'):
            encoder_cells = []
            for hidden_size in hidden_states:
                encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))

            self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)

        # run the encoder
        with tf.variable_scope('encoder_run'):
            initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_encoder, self.inputs,
                                               initial_state=initial_state,
                                               time_major=False)
            encoded_vector = self.print_tensor_with_shape(rnn_outputs[:, -1, :], "encoded")

        # the computational step for running the model
        self.encoded_vector = encoded_vector
