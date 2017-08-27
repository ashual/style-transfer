import tensorflow as tf
from v1_embedding.base_model import BaseModel


class ContentDiscriminator(BaseModel):
    def __init__(self, content_vector_size, dense_hidden_sizes, is_w_loss, dropout_placeholder, name=None):
        BaseModel.__init__(self, name)
        # discriminator - model
        self.sizes = [content_vector_size] + dense_hidden_sizes + [1]
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

    def predict(self, inputs):
        with tf.variable_scope('{}/run'.format(self.name), reuse=self.reuse_flag):
            self.reuse_flag = True
            current = inputs
            for i in range(len(self.w)):
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
