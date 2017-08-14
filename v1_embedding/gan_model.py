import tensorflow as tf

from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.loss_handler import LossHandler


# this model tries to transfer from one domain to another.
# 1. the encoder doesn't know the domain it is working on
# 2. target are encoded and decoded (to target) reconstruction is applied between the origin and the result
# 3. source is encoded decoded to target and encoded again, then L2 loss is applied between the context vectors.
# 4. an adversarial component is trained to distinguish between target inputs and source inputs
class GanModel:
    def __init__(self, config_file, operational_config_file, embedding_handler):
        self.config = config_file
        self.operational_config = operational_config_file
        self.embedding_handler = embedding_handler

        # placeholders for dropouts
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_placeholder')
        self.discriminator_dropout_placeholder = tf.placeholder(tf.float32, shape=(),
                                                                name='discriminator_dropout_placeholder')
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the left
        self.left_padded_source_batch = tf.placeholder(tf.int64, shape=(None, None), name='left_padded_source_batch')
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the right
        self.right_padded_source_batch = tf.placeholder(tf.int64, shape=(None, None), name='right_padded_source_batch')
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the left
        self.left_padded_target_batch = tf.placeholder(tf.int64, shape=(None, None), name='left_padded_target_batch')
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the right
        self.right_padded_target_batch = tf.placeholder(tf.int64, shape=(None, None), name='right_padded_target_batch')

        self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                        self.config['model']['translation_hidden_size'],
                                                        self.config['embedding']['should_train'],
                                                        self.dropout_placeholder)
        self.encoder = EmbeddingEncoder(self.config['model']['encoder_hidden_states'],
                                        self.dropout_placeholder,
                                        self.config['model']['bidirectional_encoder'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['model']['decoder_hidden_states'],
                                        self.dropout_placeholder,
                                        self.config['sentence']['max_length'])
        self.loss_handler = LossHandler(self.embedding_handler.get_vocabulary_length())
        self.discriminator = None

        # losses and accuracy:
        self.discriminator_step_prediction = None
        self.discriminator_loss = None
        self.discriminator_accuracy_for_discriminator = None

        self.generator_step_prediction = None
        self.generator_loss = None
        self.discriminator_accuracy_for_generator = None
        self.discriminator_loss_on_generator_step = None

        # train steps
        self.discriminator_train_step = None
        self.generator_train_step = None

        # do transfer
        with tf.variable_scope('TransferSourceToTarget'):
            transferred_embeddings = self._transfer(self.left_padded_source_batch)
            transferred_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                transferred_embeddings)
            self.transfer = self.embedding_translator.translate_logits_to_words(transferred_logits)

    def _encode(self, left_padded_input):
        embedding = self.embedding_translator.embed_inputs(left_padded_input)
        return self.encoder.encode_inputs_to_vector(embedding, domain_identifier=None)

    def _transfer(self, left_padded_source):
        encoded_source = self._encode(left_padded_source)
        return self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)
