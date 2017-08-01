import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingDecoder(BaseModel):
    def __init__(self, embedding_size, hidden_states, embedding_translator, dropout_placeholder, name=None):
        BaseModel.__init__(self, name)
        self.embedding_translator = embedding_translator
        # decoder - model
        with tf.variable_scope('{}/cells'.format(self.name)):
            decoder_cells = []
            for hidden_size in hidden_states:
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
                decoder_cells.append(cell)
            cell = tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout_placeholder)
            decoder_cells.append(cell)
            self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            # contains the signal for the decoder to start
            self.starting_input = tf.zeros((1, 1, embedding_size))

    def get_zero_state(self, batch_size):
        with tf.variable_scope('{}/get_zero_state'.format(self.name)):
            return self.multilayer_decoder.zero_state(batch_size, tf.float32)

    def decode_vector_to_sequence(self, encoded_vector, initial_decoder_state, inputs, domain_identifier):
        with tf.variable_scope('{}/preprocessing'.format(self.name)):
            # encoded vector (batch, context)
            encoded_vector = self.print_tensor_with_shape(encoded_vector, "encoded_vector")
            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, "inputs")

            domain_identifier = self.print_tensor_with_shape(domain_identifier, "domain_identifier")
            # initial_decoder_state = self.print_tensor_with_shape(initial_decoder_state, "initial_decoder_state")

            # important sizes
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # the decoder input need to append encoded vector to embedding of each input
            decoder_inputs = tf.expand_dims(encoded_vector, 1)
            decoder_inputs = tf.tile(decoder_inputs, [1, sentence_length, 1])
            decoder_inputs = tf.concat((inputs, decoder_inputs), axis=2)
            domain_identifier_tiled = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.expand_dims(domain_identifier, 0), 0), 0),
                [batch_size, sentence_length, 1]
            )
            decoder_inputs = tf.concat((decoder_inputs, domain_identifier_tiled), axis=2)
            decoder_inputs = self.print_tensor_with_shape(decoder_inputs, "decoder_inputs")

        with tf.variable_scope('{}/run'.format(self.name)):
            decoded_vector, decoder_last_state = tf.nn.dynamic_rnn(self.multilayer_decoder, decoder_inputs,
                                                                   initial_state=initial_decoder_state,
                                                                   time_major=False)
            decoded_vector = self.print_tensor_with_shape(decoded_vector, "decoded_vector")
            # decoder_last_state = self.print_tensor_with_shape(decoder_last_state, "decoder_last_state")

            return decoded_vector, decoder_last_state

    def do_teacher_forcing(self, encoded_vector, inputs, domain_identifier):
        with tf.variable_scope('{}/teacher_forcing'.format(self.name)):
            batch_size = tf.shape(inputs)[0]
            zero_state = self.get_zero_state(batch_size)
            starting_inputs = tf.tile(self.starting_input, (batch_size, 1, 1))
            decoder_inputs = tf.concat((starting_inputs, inputs), axis=1)
            return self.decode_vector_to_sequence(encoded_vector, zero_state, decoder_inputs, domain_identifier)[0]

    def do_iterative_decoding(self, encoded_vector, domain_identifier, iterations_limit=-1):
        # get end of sentence index
        embedding_handler = self.embedding_translator.embedding_handler
        end_index = embedding_handler.word_to_index[embedding_handler.end_of_sentence_token]

        # functions used by the while loop below.
        # common variables:
        # iteration: counter of the number of iterations
        # input_logits: logits (over the vocabulary) of the last outputted word
        # state: the sate to start the decoding from
        # inputs_from_start: all the inputs (as embedding vectors) so far
        def _while_cond(iteration, input_logits, state, inputs_from_start):
            if iterations_limit == -1:
                # make sure we are not in the first iteration:
                first_step = tf.equal(iteration, 0)
                # single sentence running until end of sentence encountered (used for test time)
                translated_index = self.embedding_translator.translate_logits_to_words(input_logits)[0][0]
                is_end_token = tf.not_equal(translated_index, end_index)
                # if first step - run loop. otherwise run if not end token
                return tf.cond(first_step, lambda: True, is_end_token)
            # not a single sentence, stopping when sentence length reached (used for professor forcing)
            return tf.less(iteration, iterations_limit)

        def _while_body(iteration, input_logits, state, inputs_from_start):
            iteration += 1
            last_input = inputs_from_start[:, -1, :]
            decoded_vector, decoder_last_state = self.decode_vector_to_sequence(encoded_vector, state, last_input,
                                                                                domain_identifier)
            inputs_from_start = tf.concat((inputs_from_start, decoded_vector), axis=1)
            # translate to logits
            input_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(decoded_vector)
            return [iteration_counter, input_logits, decoder_last_state, inputs_from_start]

        with tf.variable_scope('{}/iterative_decoding'.format(self.name)):
            batch_size = tf.shape(encoded_vector)[0]
            current_state = self.get_zero_state(batch_size)
            # get the start token and it's embedding
            current_logits = tf.zeros((1, 1, embedding_handler.get_vocabulary_length()))
            # tile both to be batch X sentence len
            current_logits = tf.tile(current_logits, [batch_size, 1, 1])
            all_inputs = tf.tile(self.starting_input, [batch_size, 1, 1])
            # desired shape for all inputs
            all_inputs_shape = all_inputs.get_shape().as_list()
            all_inputs_shape[1] = tf.Dimension(None)
            all_inputs_shape_invariant = tf.TensorShape(all_inputs_shape)
            # do the while loop
            iteration_counter = tf.Variable(0, trainable=False)
            _,_,_, all_inputs = tf.while_loop(
                # while cond
                _while_cond,
                # while body
                _while_body,
                # loop variables:
                [iteration_counter, current_logits, current_state, all_inputs],
                # shape invariants
                shape_invariants=[
                    iteration_counter.get_shape(),
                    current_logits.get_shape(),
                    current_state.get_shape(),
                    all_inputs_shape_invariant
                ],
                parallel_iterations=1,
                back_prop=True
            )
            return all_inputs
