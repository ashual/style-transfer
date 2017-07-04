import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingTranslator(BaseModel):
    def __init__(self, embedding_size, vocabulary_size, translation_hidden_size, start_token_index, stop_token_index,
                 unknown_token_index, pad_token_index):
        self.start_token_index = start_token_index
        self.stop_token_index = stop_token_index
        self.unknown_token_index = unknown_token_index
        self.pad_token_index = pad_token_index
        self.vocabulary_size = vocabulary_size

        # placeholders
        # placeholder to initiate the embedding weights
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[vocabulary_size, embedding_size])
        # placeholder to translate inputs to embedding vectors (batch, time)=> index of word
        self.inputs = tf.placeholder(tf.int32, shape=(None, None))
        # placeholder to translate embedding vector to words
        self.linear_layer_input = tf.placeholder(tf.int32, shape=(None, None, embedding_size))

        with tf.variable_scope('embedding_parameters'):
            # embedding
            self.w = tf.Variable(tf.random_normal(shape=[vocabulary_size, embedding_size]),
                                 trainable=False, name="word_vectors")

            # weights to translate embedding to vocabulary
            self.w1, self.b1 = BaseModel.create_input_parameters(embedding_size, translation_hidden_size)
            self.w2, self.b2 = BaseModel.create_input_parameters(translation_hidden_size, vocabulary_size)

    def assign_embedding(self):
        with tf.variable_scope('assign_embedding'):
            embedding = self.print_tensor_with_shape(self.embedding_placeholder, "embedding")
            return tf.assign(self.w, embedding)

    def embed_inputs(self):
        with tf.variable_scope('words_to_embeddings'):
            inputs = self.print_tensor_with_shape(self.inputs, "inputs")
            # to get vocabulary indices to embeddings
            embedded_inputs = tf.nn.embedding_lookup(self.w, inputs)
            return self.print_tensor_with_shape(embedded_inputs, "embedded_inputs")

    def translate_embedding_to_vocabulary_logits(self, inputs):
        with tf.variable_scope('embedding_to_vocabulary_logits'):
            inputs = self.print_tensor_with_shape(inputs, "inputs")
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]
            embedded_inputs_flattened = tf.reshape(inputs, (batch_size * sentence_length, -1))

            hidden = tf.nn.relu(tf.matmul(embedded_inputs_flattened, self.w1) + self.b1)
            vocabulary_logits_flattened = tf.nn.relu(tf.matmul(hidden, self.w2) + self.b2)
            vocabulary_logits = tf.reshape(vocabulary_logits_flattened, (batch_size, sentence_length, -1))
            return self.print_tensor_with_shape(vocabulary_logits, "vocabulary_logits")

    def translate_logits_to_words(self, logits_vector):
        with tf.variable_scope('vocabulary_logits_to_words'):
            actual_words = tf.argmax(logits_vector, axis=2)
            return self.print_tensor_with_shape(actual_words, "actual_words")

    def get_special_word(self, word_index):
        with tf.variable_scope('get_special_word'):
            return tf.one_hot(word_index, self.vocabulary_size)

    def is_special_word(self, word_index, logits_vector):
        with tf.variable_scope('is_special_word'):
            word_argmax = self.translate_logits_to_words(logits_vector)
            return tf.equal(word_argmax, word_index)
