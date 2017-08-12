import tensorflow as tf
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_encoder import EmbeddingEncoder


class EmbeddingDiscriminator(BaseModel):
    def __init__(self, hidden_states, dense_hidden_size, dropout_placeholder, bidirectional, name=None):
        BaseModel.__init__(self, name)
        self.encoder = EmbeddingEncoder(hidden_states, dropout_placeholder, bidirectional, name=self.name)
        # discriminator - model
        self.dense_hidden_size = dense_hidden_size
        with tf.variable_scope('{}/parameters'.format(self.name)):
            if dense_hidden_size > 0:
                self.w1, self.b1 = BaseModel.create_input_parameters(hidden_states[-1], dense_hidden_size)
                self.w2, self.b2 = BaseModel.create_input_parameters(dense_hidden_size, 1)
            else:
                self.w1, self.b1 = None, None
                self.w2, self.b2 = BaseModel.create_input_parameters(hidden_states[-1], 1)
        self.reuse_flag = False

    def predict(self, inputs):
        with tf.variable_scope('{}/run'.format(self.name), reuse=self.reuse_flag):
            self.reuse_flag = True
            rnn_res = self.encoder.encode_inputs_to_vector(inputs, None)
            rnn_res = self.print_tensor_with_shape(rnn_res, "rnn_res")
            if self.dense_hidden_size > 0:
                hidden = tf.nn.relu(tf.matmul(rnn_res, self.w1) + self.b1)
            else:
                hidden = rnn_res
            batch_norm_activations = tf.contrib.layers.batch_norm(tf.matmul(hidden, self.w2) + self.b2,
                                                                  center=True, scale=True, is_training=True)
            # batch_norm_activations = tf.contrib.layers.batch_norm(tf.matmul(hidden, self.w2) + self.b2,
            #                                                       center=True, scale=True, is_training=True,
            #                                                       scope="batch_norm", reuse=self.reuse_flag)
            prediction = tf.nn.sigmoid(batch_norm_activations)
            return self.print_tensor_with_shape(prediction, "prediction")
