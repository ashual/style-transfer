import tensorflow as tf
from v1_embedding.base_model import BaseModel


class ContentDiscriminator(BaseModel):
    def __init__(self, content_vector_size, dense_hidden_sizes, dropout_placeholder, name=None):
        BaseModel.__init__(self, name)
        # discriminator - model
        self.sizes = [content_vector_size] + dense_hidden_sizes + [1]
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
                dropout = tf.nn.dropout(current, self.dropout_placeholder)
                hidden = tf.nn.relu(tf.matmul(dropout, self.w[i]) + self.b[i])
                current = tf.contrib.layers.batch_norm(hidden, center=True, scale=True, is_training=True)
            prediction = tf.nn.sigmoid(current)
            return self.print_tensor_with_shape(prediction, "prediction")
