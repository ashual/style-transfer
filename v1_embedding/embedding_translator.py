import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingTranslator(BaseModel):
    def __init__(self, embedding_handler, translation_hidden_size, train_embeddings, dropout_placeholder, name=None):
        BaseModel.__init__(self, name)
        self.embedding_handler = embedding_handler

        # placeholders
        # placeholder to initiate the embedding weights
        embedding_shape = [embedding_handler.get_vocabulary_length(), embedding_handler.get_embedding_size()]
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=embedding_shape)
        # placeholder for the dropout
        self.dropout_placeholder = dropout_placeholder

        with tf.variable_scope('{}/parameters'.format(self.name)):
            # embedding
            self.w = tf.Variable(tf.random_normal(shape=embedding_shape),
                                 trainable=train_embeddings, name="word_vectors")
            self.extended_w = tf.concat((self.w, tf.zeros((1, embedding_shape[1]))), axis=0)

            # weights to translate embedding to vocabulary
            if translation_hidden_size == 0:
                # just a linear projection
                self.w1, self.b1 = BaseModel.create_input_parameters(embedding_shape[1], embedding_shape[0])
                self.w2, self.b2 = None, None
            else:
                # use a hidden layer
                self.w1, self.b1 = BaseModel.create_input_parameters(embedding_shape[1], translation_hidden_size)
                self.w2, self.b2 = BaseModel.create_input_parameters(translation_hidden_size, embedding_shape[0])

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

    def translate_embedding_to_vocabulary_logits(self, inputs):
        with tf.variable_scope('{}/embedding_to_vocabulary_logits'.format(self.name)):
            inputs = self.print_tensor_with_shape(inputs, "inputs")
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]
            embedded_inputs_flattened = tf.reshape(inputs, (batch_size * sentence_length, -1))

            hidden = tf.nn.relu(tf.matmul(embedded_inputs_flattened, self.w1) + self.b1)
            hidden_with_dropout = tf.nn.dropout(hidden, self.dropout_placeholder)
            vocabulary_logits_flattened = tf.nn.relu(tf.matmul(hidden_with_dropout, self.w2) + self.b2)
            vocabulary_logits = tf.reshape(vocabulary_logits_flattened, (batch_size, sentence_length, -1))
            return self.print_tensor_with_shape(vocabulary_logits, "vocabulary_logits")

    def translate_logits_to_words(self, logits_vector):
        with tf.variable_scope('{}/vocabulary_logits_to_words'.format(self.name)):
            actual_words = tf.argmax(logits_vector, axis=2)
            return self.print_tensor_with_shape(actual_words, "actual_words")

    def get_special_word(self, word_index):
        with tf.variable_scope('{}/get_special_word'.format(self.name)):
            return tf.one_hot(word_index, self.embedding_handler.get_vocabulary_length())
