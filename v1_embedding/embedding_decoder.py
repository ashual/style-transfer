import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingDecoder(BaseModel):
    def __init__(self, embedding_size, hidden_states, embedding_translator):
        BaseModel.__init__(self)
        self.embedding_translator = embedding_translator

        # decoder - model
        with tf.variable_scope('decoder', initializer=tf.random_uniform_initializer(-0.008, 0.008)):
            decoder_cells = []
            for hidden_size in hidden_states:
                decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True))
            decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True))
            self.multilayer_decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells)

    def get_trainable_parameters(self):
        return [v for v in tf.trainable_variables() if v.name.startswith('decoder_run')]

    def get_zero_state(self, batch_size):
        with tf.variable_scope('decoder_get_zero_state'):
            return self.multilayer_decoder.zero_state(batch_size, tf.float32)

    def decode_vector_to_sequence(self, encoded_vector, initial_decoder_state, inputs, domain_identifier):
        with tf.variable_scope('decoder_preprocessing'):
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

        with tf.variable_scope('decoder_run'):
            decoded_vector, decoder_last_state = tf.nn.dynamic_rnn(self.multilayer_decoder, decoder_inputs,
                                                                   initial_state=initial_decoder_state,
                                                                   time_major=False)
            decoded_vector = self.print_tensor_with_shape(decoded_vector, "decoded_vector")
            # decoder_last_state = self.print_tensor_with_shape(decoder_last_state, "decoder_last_state")

            return decoded_vector, decoder_last_state

    def do_teacher_forcing(self, encoded_vector, inputs, domain_identifier):
        zero_state = self.get_zero_state(tf.shape(inputs)[0])
        result = self.decode_vector_to_sequence(encoded_vector, zero_state, inputs, domain_identifier)[0]
        result = tf.concat((inputs[:, 0:1, :], result), axis=1)
        return result

    def do_iterative_decoding(self, encoded_vector, domain_identifier, iterations_limit=-1):
        def _while_cond(iteration_counter, input, state, inputs_from_start):
            if iterations_limit == -1:
                return tf.not_equal(input, self.embedding_translator.is_special_word(
                    self.embedding_translator.end_token_index)[0]),
            return tf.less(iteration_counter, iterations_limit)

        def _while_body(iteration_counter, input, state, inputs_from_start):
            iteration_counter += 1
            decoded_vector, decoder_last_state = self.decode_vector_to_sequence(encoded_vector, state, input,
                                                                                domain_identifier)
            inputs_from_start = tf.concat((inputs_from_start, decoded_vector), axis=0)
            # translate to logits
            input_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(decoded_vector)
            return [iteration_counter, input_logits, decoder_last_state, inputs_from_start]

        batch_size = tf.shape(encoded_vector)[0]
        current_state = self.get_zero_state(batch_size)
        # Assigning current_input the start token
        current_input = self.embedding_translator.get_special_word(self.embedding_translator.start_token_index)
        all_inputs = current_input
        # desired shape for all inputs
        all_inputs_shape = all_inputs.get_shape().as_list()
        all_inputs_shape[0] = tf.Dimension(None)
        all_inputs_shape_invariant = tf.TensorShape(all_inputs_shape)
        # iteration counter
        iteration_counter = tf.Variable(0, trainable=False)
        _,_,_, all_inputs = tf.while_loop(
            # while cond
            _while_cond,
            # while body
            _while_body,
            # loop variables:
            [iteration_counter, current_input, current_state, all_inputs],
            # shape invariants
            shape_invariants=[
                iteration_counter.get_shape(),
                current_input.get_shape(),
                current_state.get_shape(),
                all_inputs_shape_invariant
            ],
            parallel_iterations=1,
            back_prop=True
        )
        return all_inputs
