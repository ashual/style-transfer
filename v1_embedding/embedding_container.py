import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingContainer(BaseModel):
    def __init__(self, embedding_handler, train_embeddings, name=None):
        BaseModel.__init__(self, name)
        # placeholder to initiate the embedding weights
        embedding_shape = [embedding_handler.get_vocabulary_length(), embedding_handler.get_embedding_size()]
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=embedding_shape)

        with tf.variable_scope('{}/parameters'.format(self.name)):
            # embedding
            self.w = tf.Variable(tf.random_normal(shape=embedding_shape),
                                 trainable=train_embeddings, name="word_vectors")
            self.extended_w = tf.concat((self.w, tf.zeros((1, embedding_shape[1]))), axis=0)

    def assign_embedding(self):
        with tf.variable_scope('{}/assign_embeddings'.format(self.name)):
            embedding = self.print_tensor_with_shape(self.embedding_placeholder, "embedding")
            return tf.assign(self.w, embedding)

    def embed_inputs(self, inputs):
        with tf.variable_scope('{}/words_to_embeddings'.format(self.name)):
            inputs = self.print_tensor_with_shape(inputs, "inputs")
            # to get vocabulary indices to embeddings
            embedded_inputs = tf.nn.embedding_lookup(self.extended_w, inputs)
            return self.print_tensor_with_shape(embedded_inputs, "embedded_inputs")
