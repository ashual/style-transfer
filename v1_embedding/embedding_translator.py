import tensorflow as tf
from v1_embedding.base_model import BaseModel


class EmbeddingTranslator(BaseModel):
    """
    has 3 functions:
    1. use embedding_placeholder and assign_embedding to set the embedding to a specific matrix
    2. use inputs and call embedded_inputs to translate between word indices to embeddings
    3. use linear_layer_input and call vocabulary_logits to translate embedding vectors to logits of words
    4. use linear_layer_input and call actual_words to translate embedding vectors to word indices
    """
    def __init__(self, embedding_size, vocabulary_size, translation_hidden_size):
        # placeholders
        # placeholder to initiate the embedding weights
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[vocabulary_size, embedding_size])
        embedding = self.print_tensor_with_shape(self.embedding_placeholder, "embedding")
        # placeholder to translate inputs to embedding vectors (batch, time)=> index of word
        self.inputs = tf.placeholder(tf.int32, shape=(None, None))
        inputs = self.print_tensor_with_shape(self.inputs, "inputs")
        # placeholder to translate embedding vector to words
        self.linear_layer_input = tf.placeholder(tf.int32, shape=(None, None, embedding_size))
        linear_layer_input = self.print_tensor_with_shape(self.linear_layer_input, "linear_layer_input")

        # embedding
        w = tf.Variable(tf.random_normal(shape=[vocabulary_size, embedding_size]),
                        trainable=False, name="word_vectors")

        # you can call assign embedding op to init the embedding
        self.assign_embedding = tf.assign(w, embedding)
        self.w = w
        w = self.print_tensor_with_shape(self.w, "w")

        # to get vocabulary indices to embeddings
        embedded_inputs = tf.nn.embedding_lookup(w, inputs)
        self.embedded_inputs = self.print_tensor_with_shape(embedded_inputs, "embedded_inputs")

        # to get embedding vectors to vocabulary logits:
        batch_size = tf.shape(linear_layer_input)[0]
        sentence_length = tf.shape(linear_layer_input)[1]
        embedded_inputs_flattened = tf.reshape(linear_layer_input, (batch_size*sentence_length, embedding_size))

        hidden = tf.contrib.layers.fully_connected(embedded_inputs_flattened, translation_hidden_size)
        vocabulary_logits_flattened = tf.contrib.layers.fully_connected(hidden, vocabulary_size)
        vocabulary_logits = tf.reshape(vocabulary_logits_flattened, (batch_size, sentence_length, vocabulary_size))
        self.vocabulary_logits = self.print_tensor_with_shape(vocabulary_logits, "vocabulary_logits")

        # to get actual vocabulary indices from the logits
        actual_words = tf.argmax(vocabulary_logits, axis=2)
        self.actual_words = self.print_tensor_with_shape(actual_words, "actual_words")
