import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingDecoder(BaseModel):
    def __init__(self, embedding_size, hidden_states, dropout_placeholder, maximal_decoding, cell_type, name=None):
        BaseModel.__init__(self, name)
        self.maximal_decoding = maximal_decoding
        # decoder - model
        with tf.variable_scope('{}/cells'.format(self.name)):
            decoder_cells = []
            for hidden_size in hidden_states:
                if cell_type == 'GRU':
                    cell = tf.contrib.rnn.GRUCell(hidden_size)
                elif cell_type == 'LSTM':
                    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
                else:
                    raise Exception('No cell type exists')
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
                decoder_cells.append(cell)
            if cell_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(embedding_size)
            elif cell_type == 'LSTM':
                cell = tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True)
            else:
                raise Exception('No cell type exists')
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
            decoder_cells.append(cell)
            self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            # contains the signal for the decoder to start
            self.starting_input = tf.zeros((1, 1, embedding_size))
        self.reuse_flag = False

    def get_zero_state(self, batch_size):
        with tf.variable_scope('{}/get_zero_state'.format(self.name)):
            return self.multilayer_decoder.zero_state(batch_size, tf.float32)

    def decode_vector_to_sequence(self, encoded_vector, initial_decoder_state, inputs, input_lengths,
                                  domain_identifier):
            # encoded vector: (batch, context)
            # the input sequence s.t (batch, time, embedding)
        with tf.variable_scope('{}/preprocessing'.format(self.name)):
            # important sizes
            sentence_length = tf.shape(inputs)[1]
            # the decoder input need to append encoded vector to embedding of each input
            decoder_inputs = tf.expand_dims(encoded_vector, 1)
            decoder_inputs = tf.tile(decoder_inputs, [1, sentence_length, 1])
            decoder_inputs = tf.concat((inputs, decoder_inputs), axis=2)

            decoder_inputs = self.concat_identifier(decoder_inputs, domain_identifier)

        with tf.variable_scope('{}/run'.format(self.name), reuse=self.reuse_flag):
            self.reuse_flag = True
            decoded_vector, decoder_last_state = tf.nn.dynamic_rnn(self.multilayer_decoder, decoder_inputs,
                                                                   initial_state=initial_decoder_state,
                                                                   time_major=False, sequence_length=input_lengths)
            return decoded_vector, decoder_last_state

    def do_teacher_forcing(self, encoded_vector, inputs, input_lengths, domain_identifier=None):
        with tf.variable_scope('{}/teacher_forcing'.format(self.name)):
            batch_size = tf.shape(inputs)[0]
            zero_state = self.get_zero_state(batch_size)
            starting_inputs = tf.tile(self.starting_input, (batch_size, 1, 1))
            decoder_inputs = tf.concat((starting_inputs, inputs), axis=1)
            return self.decode_vector_to_sequence(encoded_vector, zero_state, decoder_inputs, input_lengths,
                                                  domain_identifier)[0]

    def do_iterative_decoding(self, encoded_vector, domain_identifier=None):
        with tf.variable_scope('{}/iterative_decoding'.format(self.name)):
            batch_size = tf.shape(encoded_vector)[0]
            # gets the initial state, and then decodes it to a single tensor to be used by the while loop
            current_state = self.get_zero_state(batch_size)
            current_input = tf.tile(self.starting_input, [batch_size, 1, 1])
            decoded_res = []
            for i in range(self.maximal_decoding):
                decoded_vector, current_state = self.decode_vector_to_sequence(
                    encoded_vector, current_state, current_input, None, domain_identifier
                )
                decoded_res.append(decoded_vector)
            return tf.concat(decoded_res, axis=1)
