import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingTranslator(BaseModel):
    def __init__(self, embedding_handler, translation_hidden_size, dropout_placeholder, name=None):
        BaseModel.__init__(self, name)
        self.vocabulary_length = embedding_handler.get_vocabulary_length()
        self.embedding_size = embedding_handler.get_embedding_size()

        # placeholders
        # placeholder to initiate the embedding weights
        embedding_shape = [self.vocabulary_length, self.embedding_size]
        # placeholder for the dropout
        self.dropout_placeholder = dropout_placeholder

        with tf.variable_scope('{}/parameters'.format(self.name)):
            # weights to translate embedding to vocabulary
            if translation_hidden_size == 0:
                # just a linear projection
                self.w1, self.b1 = None, None
                self.w2, self.b2 = BaseModel.create_input_parameters(embedding_shape[1], embedding_shape[0])
            else:
                # use a hidden layer
                self.w1, self.b1 = BaseModel.create_input_parameters(embedding_shape[1], translation_hidden_size)
                self.w2, self.b2 = BaseModel.create_input_parameters(translation_hidden_size, embedding_shape[0])

    def translate_embedding_to_vocabulary_logits(self, inputs):
        with tf.variable_scope('{}/embedding_to_vocabulary_logits'.format(self.name)):
            inputs = self.print_tensor_with_shape(inputs, "inputs")
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]
            embedded_inputs_flattened = tf.reshape(inputs, (batch_size * sentence_length, -1))
            if self.w1 is None:
                hidden_with_dropout = embedded_inputs_flattened
            else:
                hidden = tf.nn.relu(tf.matmul(embedded_inputs_flattened, self.w1) + self.b1)
                hidden_with_dropout = tf.nn.dropout(hidden, 1. - self.dropout_placeholder)
            vocabulary_logits_flattened = tf.nn.relu(tf.matmul(hidden_with_dropout, self.w2) + self.b2)
            vocabulary_logits = tf.reshape(
                vocabulary_logits_flattened,
                (batch_size, sentence_length, self.vocabulary_length)
            )
            return self.print_tensor_with_shape(vocabulary_logits, "vocabulary_logits")

    def translate_logits_to_words(self, logits_vector):
        with tf.variable_scope('{}/vocabulary_logits_to_words'.format(self.name)):
            actual_words = tf.argmax(logits_vector, axis=2)
            return self.print_tensor_with_shape(actual_words, "actual_words")
