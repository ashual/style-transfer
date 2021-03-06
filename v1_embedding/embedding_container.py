import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingContainer(BaseModel):
    def __init__(self, embedding_handler, train_embeddings, name=None):
        BaseModel.__init__(self, name)
        self.vocabulary_length = embedding_handler.get_vocabulary_length()
        embedding_shape = [self.vocabulary_length, embedding_handler.get_embedding_size()]
        # placeholder to initiate the embedding weights
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=embedding_shape)

        with tf.variable_scope('{}/parameters'.format(self.name)):
            # embedding
            self.w = tf.Variable(tf.random_normal(shape=embedding_shape),
                                 trainable=train_embeddings, name="word_vectors")
            self.extended_w = tf.concat((self.w, tf.zeros((1, embedding_shape[1]))), axis=0)

    def assign_embedding(self):
        with tf.variable_scope('{}/assign_embeddings'.format(self.name)):
            return tf.assign(self.w, self.embedding_placeholder)

    def embed_inputs(self, inputs):
        with tf.variable_scope('{}/words_to_embeddings'.format(self.name)):
            # to get vocabulary indices to embeddings
            return tf.nn.embedding_lookup(self.extended_w, inputs)

    def get_random_words_embeddings(self, shape):
        random_words = tf.random_uniform(shape=shape,
                                         minval=0, maxval=self.vocabulary_length,
                                         dtype=tf.int32)
        return self.embed_inputs(random_words)

