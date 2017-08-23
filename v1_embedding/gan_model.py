import tensorflow as tf
import numpy as np

from v1_embedding.content_discriminator import ContentDiscriminator
from v1_embedding.embedding_container import EmbeddingContainer
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
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
        self.use_discriminator_for_generator = tf.placeholder_with_default(True, shape=())
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the right
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the right
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_lengths = tf.placeholder(tf.int32, shape=(None))
        self.target_lengths = tf.placeholder(tf.int32, shape=(None))

        self.embedding_container = EmbeddingContainer(self.embedding_handler, self.config['embedding']['should_train'])
        self.embedding_translator = self._init_translator()
        self.encoder = EmbeddingEncoder(self.config['model']['encoder_hidden_states'],
                                        self.dropout_placeholder,
                                        self.config['model']['bidirectional_encoder'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['model']['decoder_hidden_states'],
                                        self.dropout_placeholder,
                                        # TODO: when add curriculum - change to max and make the transferred source be
                                        # rolled out according to the curriculum sentence length
                                        self.config['sentence']['min_length'])
        self.loss_handler = LossHandler(self.embedding_handler.get_vocabulary_length())
        self.discriminator = self._init_discriminator()
        # common steps:
        self._train_generator = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, expected_shape=())
        self._source_embedding, self._source_encoded = self._encode(self.source_batch, self.source_lengths)
        self._target_embedding, self._target_encoded = self._encode(self.target_batch, self.target_lengths)
        self._transferred_source = self.decoder.do_iterative_decoding(self._source_encoded)
        self._teacher_forced_target = self.decoder.do_teacher_forcing(
            self._target_encoded, self._target_embedding[:, :-1, :], self.target_lengths
        )
        # discriminator prediction
        self.prediction, _source_prediction, _target_prediction = self._predict()
        # loss and accuracy
        self.discriminator_loss, self.accuracy = self.loss_handler.get_discriminator_loss(_source_prediction,
                                                                                          _target_prediction)
        # content vector reconstruction loss
        transferred_source = self.decoder.do_iterative_decoding(self._source_encoded, domain_identifier=None)
        encoded_again = self.encoder.encode_inputs_to_vector(transferred_source, None, domain_identifier=None)
        self.semantic_distance_loss = self.loss_handler.get_context_vector_distance_loss(self._source_encoded,
                                                                                         encoded_again)

        # target reconstruction loss
        self.reconstruction_loss = self._get_reconstruction_loss()

        # generator loss
        generator_loss = self.config['model']['reconstruction_coefficient'] * self.reconstruction_loss + \
                         self.config['model']['semantic_distance_coefficient'] * self.semantic_distance_loss
        self._apply_discriminator_loss_for_generator = tf.logical_and(
            self.use_discriminator_for_generator,
            tf.logical_and(
                tf.greater(self.config['model']['maximal_loss_for_discriminator'], self.discriminator_loss),
                tf.greater_equal(self.accuracy, self.config['model']['minimal_accuracy_for_discriminator'])
            )
        )
        self.generator_loss = generator_loss + tf.cond(
            pred=self._apply_discriminator_loss_for_generator,
            true_fn=lambda: -self.discriminator_loss,
            false_fn=lambda: 0.0
        )

        # train steps
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('TrainSteps'):
            with tf.variable_scope('TrainDiscriminatorSteps'):
                discriminator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
                discriminator_var_list = self.discriminator.get_trainable_parameters()
                discriminator_grads_and_vars = discriminator_optimizer.compute_gradients(
                    self.discriminator_loss, colocate_gradients_with_ops=True, var_list=discriminator_var_list
                )
                with tf.control_dependencies(update_ops + [tf.assign(self._train_generator, 0)]):
                    self.discriminator_train_step = discriminator_optimizer.apply_gradients(
                        discriminator_grads_and_vars)
            with tf.variable_scope('TrainGeneratorSteps'):
                generator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
                generator_var_list = self._get_generator_step_variables()
                generator_grads_and_vars = generator_optimizer.compute_gradients(
                    self.generator_loss,
                    colocate_gradients_with_ops=True,
                    var_list=generator_var_list
                )
                with tf.control_dependencies(update_ops + [tf.assign(self._train_generator, 1)]):
                    self.generator_train_step = generator_optimizer.apply_gradients(generator_grads_and_vars)

        # do transfer
        self.transferred_source_batch = self._transfer()
        # summaries
        self.epoch, self.epoch_placeholder, self.assign_epoch = self._create_assignable_scalar(
            'epoch', tf.int32, init_value=0
        )
        self.text_watcher = TextWatcher('original', 'transferred')
        self.discriminator_step_summaries, self.generator_step_summaries = self._create_summaries()

    def _init_discriminator(self):
        if self.config['model']['discriminator_type'] == 'embedding':
            return EmbeddingDiscriminator(self.config['discriminator_embedding']['hidden_states'],
                                          self.config['discriminator_embedding']['dense_hidden_size'],
                                          self.discriminator_dropout_placeholder,
                                          self.config['discriminator_embedding']['bidirectional'])
        if self.config['model']['discriminator_type'] == 'content':
            return ContentDiscriminator(self.config['model']['encoder_hidden_states'][-1],
                                        self.config['discriminator_content']['hidden_states'],
                                        self.discriminator_dropout_placeholder)

    def _init_translator(self):
        if self.config['model']['loss_type'] == 'cross_entropy':
            return EmbeddingTranslator(self.embedding_handler,
                                       self.config['cross_entropy_loss']['translation_hidden_size'],
                                       self.dropout_placeholder)

    def _encode(self, inputs, input_lengths):
        embedding = self.embedding_container.embed_inputs(inputs)
        encoded = self.encoder.encode_inputs_to_vector(embedding, input_lengths, domain_identifier=None)
        return embedding, encoded

    def _predict(self):
        prediction_input = None
        if self.config['model']['discriminator_type'] == 'embedding':
            sentence_length = tf.shape(self._teacher_forced_target)[1]
            transferred_source_normalized = self._transferred_source[:, :sentence_length, :]
            prediction_input = tf.concat((transferred_source_normalized, self._teacher_forced_target), axis=0)
        if self.config['model']['discriminator_type'] == 'content':
            prediction_input = tf.concat((self._source_encoded, self._target_encoded), axis=0)
        prediction = self.discriminator.predict(prediction_input)
        source_batch_size = tf.shape(self.source_batch)[0]
        source_prediction = prediction[:source_batch_size, :]
        target_prediction = prediction[source_batch_size:, :]
        return prediction, source_prediction, target_prediction

    def _transfer(self):
        if self.config['model']['loss_type'] == 'cross_entropy':
            transferred_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                self._transferred_source)
            return self.embedding_translator.translate_logits_to_words(transferred_logits)
        if self.config['model']['loss_type'] == 'margin1' or self.config['model']['loss_type'] == 'margin2':
            decoded_shape = tf.shape(self._transferred_source)

            distance_tensors = []
            for vocab_word_index in range(self.embedding_handler.get_vocabulary_length()):
                relevant_w = self.embedding_container.w[vocab_word_index, :]
                expanded_w = tf.expand_dims(tf.expand_dims(relevant_w, axis=0), axis=0)
                tiled_w = tf.tile(expanded_w, [decoded_shape[0], decoded_shape[1], 1])

                square = tf.square(self._transferred_source - tiled_w)
                per_vocab_distance = tf.reduce_sum(square, axis=-1)
                distance_tensors.append(per_vocab_distance)

            distance = tf.stack(distance_tensors, axis=-1)
            best_match = tf.argmin(distance, axis=-1)
            return best_match

    def _get_reconstruction_loss(self):
        vocabulary_length = self.embedding_handler.get_vocabulary_length()
        input_shape = tf.shape(self.target_batch)
        padding_mask = tf.not_equal(self.target_batch, vocabulary_length)
        embedded_random_words = self.embedding_container.get_random_words_embeddings(
            shape=(input_shape[0], input_shape[1], self.config['margin_loss1']['random_words_size'])
        )
        if self.config['model']['loss_type'] == 'cross_entropy':
            reconstructed_target_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                self._teacher_forced_target)
            return self.loss_handler.get_sentence_reconstruction_loss(self.target_batch, reconstructed_target_logits,
                                                                      padding_mask)
        if self.config['model']['loss_type'] == 'margin1':
            distance_loss = self.loss_handler.get_distance_loss(self._target_embedding, self._teacher_forced_target,
                                                                padding_mask)

            margin = np.floor(np.sqrt(self.config['embedding']['word_size'] * 0.25))
            margin_loss = self.loss_handler.get_margin_loss(self._teacher_forced_target, padding_mask,
                                                            embedded_random_words, margin)
            return distance_loss + margin_loss * self.config['margin_loss1']['margin_coefficient']
        if self.config['model']['loss_type'] == 'margin2':
            return self.loss_handler.get_margin_loss_v2(self._target_embedding, self._teacher_forced_target,
                                                        embedded_random_words, padding_mask,
                                                        self.config['margin_loss2']['margin'])

    def _get_generator_step_variables(self):
        result = self.encoder.get_trainable_parameters() + self.decoder.get_trainable_parameters() +  \
                 self.embedding_container.get_trainable_parameters()
        if self.config['model']['loss_type'] == 'cross_entropy':
            result += self.embedding_translator.get_trainable_parameters()
        return result

    def _create_summaries(self):
        epoch_summary = tf.summary.scalar('epoch', self.epoch)
        train_generator_summary = tf.summary.scalar('train_generator', self._train_generator)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        discriminator_step_summaries = tf.summary.merge([
            epoch_summary,
            train_generator_summary,
            accuracy_summary,
            discriminator_loss_summary,
        ])
        generator_step_summaries = tf.summary.merge([
            epoch_summary,
            train_generator_summary,
            accuracy_summary,
            discriminator_loss_summary,
            tf.summary.scalar('reconstruction_loss_on_generator_step', self.reconstruction_loss),
            tf.summary.scalar('content_vector_loss_on_generator_step', self.semantic_distance_loss),
            tf.summary.scalar('generator_loss', self.generator_loss),
            tf.summary.scalar('apply_discriminator_loss_for_generator', tf.cast(
                self._apply_discriminator_loss_for_generator, tf.int8)),
        ])
        return discriminator_step_summaries, generator_step_summaries

    @staticmethod
    def _create_assignable_scalar(name, type, init_value):
        scalar = tf.Variable(initial_value=init_value, trainable=False, name='{}_var'.format(name), dtype=type)
        placeholder = tf.placeholder(dtype=type, shape=(), name='{}_assignment_placeholder'.format(name))
        assignment_op = tf.assign(scalar, placeholder)
        return scalar, placeholder, assignment_op

