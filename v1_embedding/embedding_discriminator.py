import tensorflow as tf
from v1_embedding.base_model import BaseModel
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell


class EmbeddingDiscriminator(BaseModel):
    def __init__(self, embedding_size, hidden_states, embedding_translator):
        BaseModel.__init__(self)
        self.embedding_translator = embedding_translator

        # placeholders:
        # domain identifier
        self.domain_identifier = tf.placeholder(tf.int32, shape=())

        # decoder - model
        with tf.variable_scope('discriminator_model') as vs:
            discriminator_cells = []
            for hidden_size in hidden_states:
                discriminator_cells.append(BasicLSTMCell(hidden_size, state_is_tuple=True))
            discriminator_cells.append(BasicLSTMCell(embedding_size, state_is_tuple=True))
            self.multilayer_discriminator = MultiRNNCell(discriminator_cells)

            # Retrieve just the LSTM variables.
            self.model_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

    def decode_vector_to_sequence(self, inputs):
        with tf.variable_scope('discriminator_preprocessing'):
            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, 'inputs')

            # important sizes
            batch_size = tf.shape(inputs)[0]

        # run the discriminator
        with tf.variable_scope('discriminator_run'):
            # define the initial state as empty
            initial_state = self.multilayer_discriminator.zero_state(batch_size, tf.float32)
            # run the model
            rnn_outputs, _ = tf.nn.dynamic_rnn(self.multilayer_discriminator, inputs,
                                               initial_state=initial_state,
                                               time_major=False)
            return self.print_tensor_with_shape(rnn_outputs[:, -1, :], 'discriminator_results')

    def get_trainable_parameters(self):
        return self.model_variables