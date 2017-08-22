import tensorflow as tf
import numpy as np

from v1_embedding.embedding_container import EmbeddingContainer
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.loss_handler import LossHandler
from v1_embedding.text_watcher import TextWatcher


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
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the right
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the right
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_lengths = tf.placeholder(tf.int32, shape=(None))
        self.target_lengths = tf.placeholder(tf.int32, shape=(None))

        self.embedding_container = EmbeddingContainer(self.embedding_handler, self.config['embedding']['should_train'])
        self.embedding_translator = None
        if self.config['model']['loss_type'] == 'cross_entropy':
            self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                            self.config['model']['translation_hidden_size'],
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
        self.discriminator_loss_on_discriminator_step = None
        self.discriminator_accuracy_for_discriminator = None

        self.generator_step_prediction = None
        self.generator_loss = None
        self.discriminator_accuracy_for_generator = None
        self.discriminator_loss_on_generator_step = None
        self.reconstruction_loss_on_generator_step = None
        self.content_vector_loss_on_generator_step = None

        # train steps
        self.discriminator_train_step = None
        self.generator_train_step = None

        # do transfer
        transferred_embeddings = self._transfer(self.source_batch, self.source_lengths)
        if self.config['model']['loss_type'] == 'cross_entropy':
            transferred_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                transferred_embeddings)
            self.transfer = self.embedding_translator.translate_logits_to_words(transferred_logits)
        else:
            self.transfer = self.decoded_to_closest(transferred_embeddings,
                                                    self.embedding_handler.get_vocabulary_length())

        # summaries
        self.epoch, self.epoch_placeholder, self.assign_epoch = GanModel._create_assignable_scalar(
            'epoch', tf.int32, init_value=0
        )
        self.train_generator, self.train_generator_placeholder, self.assign_train_generator = \
            GanModel._create_assignable_scalar(
                'train_generator', tf.int32, init_value=0
        )
        self.text_watcher = TextWatcher('original', 'transferred')

    def decoded_to_closest(self, decoded, vocabulary_length):
        decoded_shape = tf.shape(decoded)

        distance_tensors = []
        for vocab_word_index in range(vocabulary_length):
            relevant_w = self.embedding_container.w[vocab_word_index, :]
            expanded_w = tf.expand_dims(tf.expand_dims(relevant_w, axis=0), axis=0)
            tiled_w = tf.tile(expanded_w, [decoded_shape[0], decoded_shape[1], 1])

            square = tf.square(decoded - tiled_w)
            per_vocab_distance = tf.reduce_sum(square, axis=-1)
            distance_tensors.append(per_vocab_distance)

        distance = tf.stack(distance_tensors, axis=-1)
        best_match = tf.argmin(distance, axis=-1)
        return best_match

    def create_summaries(self):
        epoch_summary = tf.summary.scalar('epoch', self.epoch)
        train_generator_summary = tf.summary.scalar('train_generator', self.train_generator)
        discriminator_step_summaries = tf.summary.merge([
            epoch_summary,
            train_generator_summary,
            tf.summary.scalar('discriminator_accuracy_for_discriminator', self.discriminator_accuracy_for_discriminator),
            tf.summary.scalar('discriminator_loss_on_discriminator_step', self.discriminator_loss_on_discriminator_step),
        ])
        generator_step_summaries = tf.summary.merge([
            epoch_summary,
            train_generator_summary,
            tf.summary.scalar('discriminator_accuracy_for_generator', self.discriminator_accuracy_for_generator),
            tf.summary.scalar('discriminator_loss_on_generator_step', self.discriminator_loss_on_generator_step),
            tf.summary.scalar('reconstruction_loss_on_generator_step', self.reconstruction_loss_on_generator_step),
            tf.summary.scalar('content_vector_loss_on_generator_step', self.content_vector_loss_on_generator_step),
            tf.summary.scalar('generator_loss', self.generator_loss),
        ])
        return discriminator_step_summaries, generator_step_summaries

    def _encode(self, inputs, input_lengths):
        embedding = self.embedding_container.embed_inputs(inputs)
        return self.encoder.encode_inputs_to_vector(embedding, input_lengths, domain_identifier=None)

    def _transfer(self, inputs, input_lengths):
        encoded_source = self._encode(inputs, input_lengths)
        return self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)

    def _get_generator_step_variables(self):
        result = self.encoder.get_trainable_parameters() + self.decoder.get_trainable_parameters() +  \
                 self.embedding_container.get_trainable_parameters()
        if self.config['model']['loss_type'] == 'cross_entropy':
            result += self.embedding_translator.get_trainable_parameters()
        return result

    def _reconstruction_loss(self, original_indices, decoded_embeddings):
        vocabulary_length = self.embedding_handler.get_vocabulary_length()
        input_shape = tf.shape(original_indices)
        original_embedding = self.embedding_container.embed_inputs(original_indices)
        if self.config['model']['loss_type'] == 'cross_entropy':
            reconstructed_target_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                decoded_embeddings)
            return self.loss_handler.get_sentence_reconstruction_loss(original_indices, reconstructed_target_logits)
        if self.config['model']['loss_type'] == 'margin1':
            padding_mask = tf.not_equal(original_indices, vocabulary_length)
            distance_loss = self.loss_handler.get_distance_loss(original_embedding, decoded_embeddings, padding_mask)
            embedded_random_words = self.embedding_container.get_random_words_embeddings(
                shape=(input_shape[0], input_shape[1], self.config['margin_loss1']['random_words_size'])
            )
            margin = np.floor(np.sqrt(self.config['embedding']['word_size'] * 0.25))
            margin_loss = self.loss_handler.get_margin_loss(decoded_embeddings, padding_mask, embedded_random_words,
                                                            margin)
            return distance_loss + margin_loss * self.config['margin_loss1']['margin_coefficient']
        if self.config['model']['loss_type'] == 'margin2':
            padding_mask = tf.not_equal(original_indices, vocabulary_length)
            embedded_random_words = self.embedding_container.get_random_words_embeddings(
                shape=(input_shape[0], input_shape[1], self.config['margin_loss2']['random_words_size'])
            )
            return self.loss_handler.get_margin_loss_v2(original_embedding, decoded_embeddings, embedded_random_words,
                                                        self.config['margin_loss2']['margin'], padding_mask)

    @staticmethod
    def _create_assignable_scalar(name, type, init_value):
        scalar = tf.Variable(initial_value=init_value, trainable=False, name='{}_var'.format(name), dtype=type)
        placeholder = tf.placeholder(dtype=type, shape=(), name='{}_assignment_placeholder'.format(name))
        assignment_op = tf.assign(scalar, placeholder)
        return scalar, placeholder, assignment_op

