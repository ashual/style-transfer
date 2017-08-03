import tensorflow as tf
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_encoder import EmbeddingEncoder


class EmbeddingDiscriminator(BaseModel):
    def __init__(self, hidden_states, dense_hidden_size, dropout_placeholder, bidirectional,
                 name=None):
        BaseModel.__init__(self, name)
        self.encoder = EmbeddingEncoder(hidden_states, dropout_placeholder, bidirectional, name=self.name)
        # discriminator - model
        with tf.variable_scope('{}/parameters'.format(self.name)):
            self.w1, self.b1 = BaseModel.create_input_parameters(hidden_states[-1], dense_hidden_size)
            self.w2, self.b2 = BaseModel.create_input_parameters(dense_hidden_size, 2)

    def predict(self, inputs):
        with tf.variable_scope('{}/run'.format(self.name)):
            rnn_res = self.encoder.encode_inputs_to_vector(inputs, None)
            rnn_res = self.print_tensor_with_shape(rnn_res, "rnn_res")
            hidden = tf.nn.relu(tf.matmul(rnn_res, self.w1) + self.b1)
            logits = tf.nn.relu(tf.matmul(hidden, self.w2) + self.b2)
            prediction = tf.nn.softmax(logits)
            return self.print_tensor_with_shape(prediction, "prediction")