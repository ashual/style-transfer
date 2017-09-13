import tensorflow as tf
import numpy as np

from v1_embedding.content_discriminator import ContentDiscriminator
from v1_embedding.embedding_container import EmbeddingContainer
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.iterative_policy import IterativePolicy
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

        self.epoch, self.epoch_placeholder, self.assign_epoch = self._create_assignable_scalar(
            'epoch', tf.int32, init_value=0
        )
        self._apply_discriminator_loss_for_generator_counter = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._generator_steps_counter = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._total_steps_counter = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.source_lengths = tf.placeholder(tf.int32, shape=(None))
        self.target_lengths = tf.placeholder(tf.int32, shape=(None))

        self.embedding_container = EmbeddingContainer(self.embedding_handler, self.config['embedding']['should_train'])
        self.embedding_translator = self._init_translator()
        self.encoder = EmbeddingEncoder(self.config['model']['encoder_hidden_states'],
                                        self.dropout_placeholder,
                                        self.config['model']['bidirectional_encoder'],
                                        self.config['model']['cell_type'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['model']['decoder_hidden_states'],
                                        self.dropout_placeholder,
                                        # TODO: when add curriculum - change to max and make the transferred source be
                                        # rolled out according to the curriculum sentence length
                                        self.config['sentence']['min_length'],
                                        self.config['model']['cell_type'])
        self.loss_handler = LossHandler(self.embedding_handler.get_vocabulary_length())
        self.discriminator = self._init_discriminator()
        self.policy = IterativePolicy(True, generator_steps=self.config['trainer']['min_generator_steps'],
                                      discriminator_steps=self.config['trainer']['min_discriminator_steps'])

        # common steps:
        self._source_embedding, self._source_encoded = self._encode(self.source_batch, self.source_lengths)
        self._target_embedding, self._target_encoded = self._encode(self.target_batch, self.target_lengths)
        self._transferred_source = self.decoder.do_iterative_decoding(self._source_encoded)
        self._teacher_forced_target = self.decoder.do_teacher_forcing(
            self._target_encoded, self._target_embedding[:, :-1, :], self.target_lengths
        )

        # discriminator prediction
        self.prediction, self._source_prediction, self._target_prediction = self._predict()

        # discriminator loss and accuracy
        discriminator_loss, self.accuracy = self._apply_discriminator_loss()
        self.discriminator_loss = self.config['model']['discriminator_coefficient'] * discriminator_loss

        # content vector reconstruction loss
        encoded_again = self.encoder.encode_inputs_to_vector(self._transferred_source, None, domain_identifier=None)
        self.semantic_distance_loss = self.config['model']['semantic_distance_coefficient'] * \
                                      self.loss_handler.get_context_vector_distance_loss(self._source_encoded,
                                                                                         encoded_again)

        # target reconstruction loss
        self.reconstruction_loss = self.config['model']['reconstruction_coefficient'] * self._get_reconstruction_loss()

        # generator loss
        generator_loss = self.reconstruction_loss
        # flag indicating if we are starting with just generator training
        is_initial_generator_epochs = tf.less(self.epoch, self.config['model']['initial_generator_epochs'])
        # if we are in the initial epochs just do reconstruction loss
        generator_loss = generator_loss + tf.cond(
            pred=is_initial_generator_epochs,
            true_fn=lambda: 0.0,
            false_fn=lambda: self.semantic_distance_loss,
        )
        self._apply_discriminator_loss_for_generator = tf.logical_and(
            # is discriminator well behaved
            tf.logical_and(
                tf.greater(self.config['model']['maximal_loss_for_discriminator'], self.discriminator_loss),
                tf.greater_equal(self.accuracy, self.config['model']['minimal_accuracy_for_discriminator'])
            ),
            # we are not in initial epochs
            tf.logical_not(is_initial_generator_epochs)
        )
        self.generator_loss = generator_loss + tf.cond(
            pred=self._apply_discriminator_loss_for_generator,
            true_fn=lambda: -self.discriminator_loss,
            false_fn=lambda: 0.0
        )

        # train steps
        with tf.variable_scope('TrainSteps'):
            self._discriminator_train_step = self._get_discriminator_train_step()
            self._generator_train_step = self._get_generator_train_step()
            self.train_generator = tf.logical_or(self.policy.should_train_generator(), is_initial_generator_epochs)

            # controls which train step to take
            policy_selected_train_step = tf.cond(
                self.train_generator,
                lambda: self._generator_train_step,
                lambda: self._discriminator_train_step
            )
            # steps to increase counters
            counter_steps = tf.group(
                self._increase_if(self._generator_steps_counter, self.train_generator),
                self._increase_if(self._apply_discriminator_loss_for_generator_counter,
                                  tf.logical_and(self.train_generator, self._apply_discriminator_loss_for_generator)),
                tf.assign_add(self._total_steps_counter, 1.0)
            )

            # after appropriate train step is executed, we notify the policy and count for tensorboard
            with tf.control_dependencies([policy_selected_train_step, counter_steps]):
                # this is the master step to use to run a train step on the model
                self.master_step = tf.cond(
                    # notify the policy only if we are not in the initial steps
                    pred=is_initial_generator_epochs,
                    true_fn=lambda: tf.no_op(),
                    false_fn=lambda: self.policy.notify(),
                )

        # summaries
        discriminator_step_summaries, generator_step_summaries = self._create_summaries()
        # controls which summary step to take
        self.summary_step = tf.cond(
            self.train_generator,
            lambda: generator_step_summaries,
            lambda: discriminator_step_summaries
        )

        # do transfer
        self.transferred_source_batch = self._translate_to_vocabulary(self._transferred_source)
        # reconstruction
        self.reconstructed_targets_batch = self._translate_to_vocabulary(self._teacher_forced_target)

        # to generate text in tensorboard use:
        self.text_watcher = TextWatcher(['original_source', 'original_target', 'transferred', 'reconstructed'])

    def _increase_if(self, variable, condition):
        return tf.assign_add(
            variable,
            tf.cond(
                pred=condition,
                true_fn=lambda: 1.0,
                false_fn=lambda: 0.0,
            )
        )

    def _init_discriminator(self):
        is_w_loss = self.config['model']['discriminator_loss_type'] == 'wasserstein'
        if self.config['model']['discriminator_type'] == 'embedding':
            dense_inputs = self.config['discriminator_embedding']['encoder_hidden_states'][-1]
            if self.config['discriminator_embedding']['include_content_vector']:
                dense_inputs += self.config['model']['encoder_hidden_states'][-1]
            return EmbeddingDiscriminator(self.config['discriminator_embedding']['encoder_hidden_states'],
                                          dense_inputs,
                                          self.config['discriminator_embedding']['hidden_states'],
                                          is_w_loss,
                                          self.discriminator_dropout_placeholder,
                                          self.config['discriminator_embedding']['bidirectional'],
                                          self.config['model']['cell_type'])
        if self.config['model']['discriminator_type'] == 'content':
            return ContentDiscriminator(self.config['model']['encoder_hidden_states'][-1],
                                        self.config['discriminator_content']['hidden_states'],
                                        is_w_loss,
                                        self.discriminator_dropout_placeholder)

    def _init_translator(self):
        if self.config['model']['loss_type'] == 'cross_entropy':
            return EmbeddingTranslator(self.embedding_handler,
                                       self.config['cross_entropy_loss']['translation_hidden_size'],
                                       self.dropout_placeholder)

    def _get_optimizer(self):
        learn_rate = self.config['model']['learn_rate']
        if self.config['model']['optimizer'] == 'gd':
            return tf.train.GradientDescentOptimizer(learn_rate)
        if self.config['model']['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(learn_rate)
        if self.config['model']['optimizer'] == 'rmsp':
            return tf.train.RMSPropOptimizer(learn_rate)

    def _encode(self, inputs, input_lengths):
        embedding = self.embedding_container.embed_inputs(inputs)
        encoded = self.encoder.encode_inputs_to_vector(embedding, input_lengths, domain_identifier=None)
        return embedding, encoded

    def _predict(self):
        if self.config['model']['discriminator_type'] == 'embedding':
            sentence_length = tf.shape(self._teacher_forced_target)[1]
            transferred_source_normalized = self._transferred_source[:, :sentence_length, :]
            prediction_input = tf.concat((transferred_source_normalized, self._teacher_forced_target), axis=0)
            if self.config['discriminator_embedding']['include_content_vector']:
                encoded = tf.concat((self._source_encoded, self._target_encoded), axis=0)
            else:
                encoded = None
            prediction = self.discriminator.predict(prediction_input, encoded)
        if self.config['model']['discriminator_type'] == 'content':
            prediction_input = tf.concat((self._source_encoded, self._target_encoded), axis=0)
            prediction = self.discriminator.predict(prediction_input)
        source_batch_size = tf.shape(self.source_batch)[0]
        source_prediction, target_prediction = tf.split(prediction, [source_batch_size, source_batch_size], axis=0)
        return prediction, source_prediction, target_prediction

    def _apply_discriminator_loss(self):
        source_prediction = self._source_prediction
        target_prediction = self._target_prediction
        if self.config['model']['discriminator_loss_type'] == 'regular':
            return self.loss_handler.get_discriminator_loss(source_prediction, target_prediction)
        if self.config['model']['discriminator_loss_type'] == 'wasserstein':
            return self.loss_handler.get_discriminator_loss_wasserstien(source_prediction, target_prediction)

    def _translate_to_vocabulary(self, embeddings):
        if self.config['model']['loss_type'] == 'cross_entropy':
            transferred_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
                embeddings)
            return self.embedding_translator.translate_logits_to_words(transferred_logits)
        if self.config['model']['loss_type'] == 'margin1' or self.config['model']['loss_type'] == 'margin2':
            decoded_shape = tf.shape(embeddings)

            distance_tensors = []
            for vocab_word_index in range(self.embedding_handler.get_vocabulary_length()):
                relevant_w = self.embedding_container.w[vocab_word_index, :]
                expanded_w = tf.expand_dims(tf.expand_dims(relevant_w, axis=0), axis=0)
                tiled_w = tf.tile(expanded_w, [decoded_shape[0], decoded_shape[1], 1])

                square = tf.square(embeddings - tiled_w)
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

    def _get_discriminator_train_step(self):
        # the embedding discriminator has batch norm that we need to update
        update_ops = None
        if self.config['model']['discriminator_type'] == 'embedding':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('TrainDiscriminatorSteps'):
            discriminator_optimizer = self._get_optimizer()
            discriminator_var_list = self.discriminator.get_trainable_parameters()

            discriminator_grads_and_vars = discriminator_optimizer.compute_gradients(
                self.discriminator_loss, colocate_gradients_with_ops=True, var_list=discriminator_var_list
            )
            if update_ops is None:
                discriminator_train_step = discriminator_optimizer.apply_gradients(discriminator_grads_and_vars)
            else:
                with tf.control_dependencies(update_ops):
                    discriminator_train_step = discriminator_optimizer.apply_gradients(discriminator_grads_and_vars)
            if self.config['model']['discriminator_loss_type'] == 'regular':
                return discriminator_train_step
            if self.config['model']['discriminator_loss_type'] == 'wasserstein':
                clip_val = self.config['wasserstein_loss']['clip_value']
                clip_discriminator = [p.assign(tf.clip_by_value(p, -clip_val, clip_val)) for p in discriminator_var_list]
                with tf.control_dependencies([discriminator_train_step]):
                    discriminator_train_step = tf.group(*clip_discriminator)
                return discriminator_train_step

    def _get_generator_train_step(self):
        with tf.variable_scope('TrainGeneratorSteps'):
            generator_optimizer = self._get_optimizer()
            generator_var_list = self._get_generator_step_variables()
            generator_grads_and_vars = generator_optimizer.compute_gradients(
                self.generator_loss,
                colocate_gradients_with_ops=True,
                var_list=generator_var_list
            )
            return generator_optimizer.apply_gradients(generator_grads_and_vars)

    def _create_ratio_summary(self, nominator, denominator):
        return tf.cond(
            pred=tf.greater(denominator, 0.0),
            true_fn=lambda: nominator / denominator,
            false_fn=lambda: 0.0
        )

    def _create_summaries(self):
        epoch_summary = tf.summary.scalar('epoch', self.epoch)
        # train_generator_summary = tf.summary.scalar('train_generator', tf.cast(self.train_generator, dtype=tf.int8)),
        train_generator_summary = tf.summary.scalar(
            'train_generator',
            self._create_ratio_summary(self._generator_steps_counter, self._total_steps_counter)
        )
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
            # tf.summary.scalar('apply_discriminator_loss_for_generator', tf.cast(
            #     self._apply_discriminator_loss_for_generator, tf.int8)),
            tf.summary.scalar(
                'apply_discriminator_loss_for_generator',
                self._create_ratio_summary(self._apply_discriminator_loss_for_generator_counter,
                                           self._generator_steps_counter)
            )
        ])
        return discriminator_step_summaries, generator_step_summaries

    @staticmethod
    def _create_assignable_scalar(name, type, init_value):
        scalar = tf.Variable(initial_value=init_value, trainable=False, name='{}_var'.format(name), dtype=type)
        placeholder = tf.placeholder(dtype=type, shape=(), name='{}_assignment_placeholder'.format(name))
        assignment_op = tf.assign(scalar, placeholder)
        return scalar, placeholder, assignment_op

