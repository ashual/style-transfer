import tensorflow as tf
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_encoder import EmbeddingEncoder


class EmbeddingDiscriminator(BaseModel):
    def __init__(self, encoder_hidden_states, dense_inputs, dense_hidden_states, is_w_loss, dropout_placeholder, bidirectional, name=None):
        BaseModel.__init__(self, name)
        self.encoder = EmbeddingEncoder(0, False, encoder_hidden_states, dropout_placeholder, bidirectional, name=self.name)
        self.sizes = [dense_inputs] + dense_hidden_states + [1]
        self.is_w_loss = is_w_loss
        self.dropout_placeholder = dropout_placeholder
        self.w = []
        self.b = []
        with tf.variable_scope('{}/parameters'.format(self.name)):
            for i in range(len(self.sizes)-1):
                w, b = BaseModel.create_input_parameters(self.sizes[i], self.sizes[i+1])
                self.w.append(w)
                self.b.append(b)
        self.reuse_flag = False

    def predict(self, inputs, encoded_vector=None):
        with tf.variable_scope('{}/run'.format(self.name), reuse=self.reuse_flag):
            self.reuse_flag = True
            rnn_res = self.encoder.encode_inputs_to_vector(inputs, None)
            if encoded_vector is not None:
                current = tf.concat((rnn_res, encoded_vector), axis=1)
            else:
                current = rnn_res
            for i in range(len(self.w)):
                current = tf.nn.dropout(current, 1.0 - self.dropout_placeholder)
                # if w loss use relu
                if self.is_w_loss:
                    current = tf.matmul(current, self.w[i]) + self.b[i]
                    if i < len(self.w) - 1:
                        current = tf.nn.relu(current)
                # else chain sigmoids
                else:
                    pre_activation = tf.matmul(current, self.w[i]) + self.b[i]
                    offset = 0.5 * tf.ones_like(pre_activation) + pre_activation
                    current = tf.nn.sigmoid(offset)
            return current

