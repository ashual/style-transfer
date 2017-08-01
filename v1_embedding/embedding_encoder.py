import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingEncoder(BaseModel):
    def __init__(self, hidden_states, dropout_placeholder, bidirectional, name=None):
        BaseModel.__init__(self, name)
        self.bidirectional = bidirectional
        # encoder - model
        with tf.variable_scope('{}/cells'.format(self.name)):
            if bidirectional:
                self.multilayer_encoder_fw = tf.contrib.rnn.MultiRNNCell(self.generate_cells(hidden_states,
                                                                                             dropout_placeholder))
                self.multilayer_encoder_bw = tf.contrib.rnn.MultiRNNCell(self.generate_cells(hidden_states,
                                                                                             dropout_placeholder))
            else:
                self.multilayer_encoder = tf.contrib.rnn.MultiRNNCell(self.generate_cells(hidden_states,
                                                                                          dropout_placeholder))
    @staticmethod
    def generate_cells(hidden_states, dropout_placeholder):
        encoder_cells = []
        for hidden_size in hidden_states:
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
            encoder_cells.append(cell)
        return encoder_cells

    def encode_inputs_to_vector(self, inputs, domain_identifier):
        with tf.variable_scope('{}/preprocessing'.format(self.name)):
            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # if domain identifier is None the encoder has no information about the source domain
            if domain_identifier is None:
                encoder_inputs = inputs
            else:
                # the input sequence s.t (batch, time, embedding)
                inputs = self.print_tensor_with_shape(inputs, "inputs")

                # create the input: (batch, time, embedding; domain)
                domain_identifier = self.print_tensor_with_shape(domain_identifier, "domain_identifier")
                domain_identifier_tiled = domain_identifier * tf.ones([batch_size, sentence_length, 1])
                encoder_inputs = tf.concat((inputs, domain_identifier_tiled), axis=2)

        # run the encoder
        with tf.variable_scope('{}/run'.format(self.name)):
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

