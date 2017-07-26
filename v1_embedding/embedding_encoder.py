import tensorflow as tf
import copy
from v1_embedding.base_model import BaseModel


class EmbeddingEncoder(BaseModel):
    def __init__(self, hidden_states, context_vector_size, dropout_placeholder, bidirectional):
        BaseModel.__init__(self)
        self.bidirectional = bidirectional
        # encoder - model
        with tf.variable_scope('encoder', initializer=tf.random_uniform_initializer(-0.008, 0.008)):
            encoder_cells = []
            for hidden_size in hidden_states:
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
                encoder_cells.append(cell)
            cell = tf.contrib.rnn.BasicLSTMCell(context_vector_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
            encoder_cells.append(cell)
            if bidirectional:
                self.multilayer_encoder_fw = tf.contrib.rnn.MultiRNNCell(copy.copy(encoder_cells))
                self.multilayer_encoder_bw = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            else:
                self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells)

    def get_trainable_parameters(self):
        return [v for v in tf.trainable_variables() if v.name.startswith('encoder_run')]

    def encode_inputs_to_vector(self, inputs, domain_identifier):
        with tf.variable_scope('encoder_preprocessing'):
            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, "inputs")

            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # create the input: (batch, time, embedding; domain)
            domain_identifier = self.print_tensor_with_shape(domain_identifier, "domain_identifier")
            domain_identifier_tiled = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(domain_identifier, 0), 0), 0),
                [batch_size, sentence_length, 1]
            )
            encoder_inputs = tf.concat((inputs, domain_identifier_tiled), axis=2)

        # run the encoder
        with tf.variable_scope('encoder_run'):
            # define the initial state as empty and run model
            if self.bidirectional:
                initial_state_bw = self.multilayer_encoder_bw.zero_state(batch_size, tf.float32)
                initial_state_fw = self.multilayer_encoder_fw.zero_state(batch_size, tf.float32)
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.multilayer_encoder_fw, self.multilayer_encoder_bw,
                                                                 encoder_inputs, initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw, time_major=False)
                res = tf.concat((rnn_outputs[0][:, -1, :], rnn_outputs[1][:, -1, :]), axis=-1)
            else:
                initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_encoder, encoder_inputs,
                                                   initial_state=initial_state,
                                                   time_major=False)
                res = rnn_outputs[:, -1, :]
            return self.print_tensor_with_shape(res, "encoded")

