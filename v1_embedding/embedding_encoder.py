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
        self.reuse_flag = False

    @staticmethod
    def generate_cells(hidden_states, dropout_placeholder):
        encoder_cells = []
        for hidden_size in hidden_states:
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
            encoder_cells.append(cell)
        return encoder_cells

    def encode_inputs_to_vector(self, inputs, input_lengths, domain_identifier=None):
        with tf.variable_scope('{}/preprocessing'.format(self.name)):
            encoder_inputs = self.concat_identifier(inputs, domain_identifier)

        # run the encoder
        with tf.variable_scope('{}/run'.format(self.name), reuse=self.reuse_flag):
            self.reuse_flag = True
            batch_size = tf.shape(inputs)[0]
            # define the initial state as empty and run model
            if self.bidirectional:
                initial_state_bw = self.multilayer_encoder_bw.zero_state(batch_size, tf.float32)
                initial_state_fw = self.multilayer_encoder_fw.zero_state(batch_size, tf.float32)
                _, final_state = tf.nn.bidirectional_dynamic_rnn(self.multilayer_encoder_fw, self.multilayer_encoder_bw,
                                                                 encoder_inputs, initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw, time_major=False,
                                                                 sequence_length=input_lengths)
                res = tf.concat((final_state[0][-1].h, final_state[1][-1].h), axis=-1)
            else:
                initial_state = self.multilayer_encoder.zero_state(batch_size, tf.float32)
                _, final_state = tf.nn.dynamic_rnn(self.multilayer_encoder, encoder_inputs,
                                                   initial_state=initial_state,
                                                   time_major=False, sequence_length=input_lengths)
                res = final_state[-1].h
            return res

