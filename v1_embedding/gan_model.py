import tensorflow as tf

from v1_embedding.content_discriminator import ContentDiscriminator
from v1_embedding.embedding_container import EmbeddingContainer
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.iterative_policy import IterativePolicy
from v1_embedding.loss_handler import LossHandler
from v1_embedding.text_watcher import TextWatcher
from v1_embedding.tf_counter import TfCounter


# this model tries to transfer from one domain to another.
# 1. the encoder doesn't know the domain it is working on
# 2. target are encoded and decoded (to target) reconstruction is applied between the origin and the result
# 3. source is encoded decoded to target and encoded again, then L2 loss is applied between the context vectors.
# 4. an adversarial component is trained to distinguish between target inputs and source inputs


class GanModel:
    def __init__(self, config_file, operational_config_file, embedding_handler):
        self.config = config_file
        self.operational_config = operational_config_file
        self.do_tensorboard = operational_config_file['tensorboard_frequency'] > 0
        self.embedding_handler = embedding_handler

        # placeholders for dropouts
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout_placeholder')
        self.discriminator_dropout_placeholder = tf.placeholder(tf.float32, shape=(),
                                                                name='discriminator_dropout_placeholder')
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the right
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the right
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))
        # epoch counter
        self.epoch_counter = TfCounter('epoch')
        # variables to store counters (only if tensorboard is activated)
        if self.do_tensorboard:
            self._apply_discriminator_loss_for_generator_counter = TfCounter('apply_discriminator_loss_for_generator')
            self._generator_steps_counter = TfCounter('generator_steps')
            self._total_steps_counter = TfCounter('total_steps')

        self.source_lengths = tf.placeholder(tf.int32, shape=(None))
        self.target_lengths = tf.placeholder(tf.int32, shape=(None))

        self.embedding_container = EmbeddingContainer(self.embedding_handler, self.config['embedding']['should_train'])
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
        self.discriminator_loss, self.accuracy = self.loss_handler.get_discriminator_loss_wasserstien(
            self._source_prediction, self._target_prediction)

        # target reconstruction loss
        self.reconstruction_loss = self._get_reconstruction_loss()

        # generator loss
        # flag indicating if we are starting with just generator training
        is_initial_generator_epochs = tf.less(self.epoch_counter.count,
                                              self.config['model']['initial_generator_epochs'])
        self._apply_discriminator_loss_for_generator = tf.cond(
            pred=is_initial_generator_epochs,
            # if initial generator epoch - return false
            true_fn=lambda: tf.constant(False),
            # else, check that discriminator is accurate
            false_fn=lambda: tf.greater_equal(self.accuracy, self.config['model']['minimal_accuracy_for_discriminator'])
        )

        self.generator_loss = self.reconstruction_loss + tf.cond(
            pred=self._apply_discriminator_loss_for_generator,
            true_fn=lambda: -self.config['model']['discriminator_coefficient'] * self.discriminator_loss,
            false_fn=lambda: 0.0
        )

        # train steps
        with tf.variable_scope('TrainSteps'):
            self._discriminator_train_step = self._get_discriminator_train_step()
            self._generator_train_step = self._get_generator_train_step()
            self.train_generator = tf.cond(
                pred=is_initial_generator_epochs,
                true_fn=lambda: tf.constant(True),
                false_fn=lambda: self.policy.should_train_generator()
            )
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
                                  tf.cond(
                                      pred=self.train_generator,
                                      true_fn=lambda: self._apply_discriminator_loss_for_generator,
                                      false_fn=lambda: tf.constant(False)
                                  )),
                self._total_steps_counter.update
            ) if self.do_tensorboard else None

            # after appropriate train step is executed, we notify the policy and count for tensorboard
            control_list = [policy_selected_train_step, counter_steps] if self.do_tensorboard else [
                policy_selected_train_step]
            with tf.control_dependencies(control_list):
                # this is the master step to use to run a train step on the model
                self.master_step = tf.cond(
                    # notify the policy only if we are not in the initial steps
                    pred=is_initial_generator_epochs,
                    true_fn=lambda: tf.no_op(),
                    false_fn=lambda: self.policy.notify(),
                )

        self.summary_step, self.text_watcher, self.evaluation_summary = None, None, None
        if self.do_tensorboard:
            # summaries
            discriminator_step_summaries, generator_step_summaries = self._create_summaries()
            # controls which summary step to take
            self.summary_step = tf.cond(
                self.train_generator,
                lambda: generator_step_summaries,
                lambda: discriminator_step_summaries
            )
            # to generate text in tensorboard use:
            self.text_watcher = TextWatcher(['original_source', 'original_target', 'transferred', 'reconstructed'])
            self.evaluation_summary = self.text_watcher.summary

        # do transfer
        self.transferred_source_batch = self._translate_to_vocabulary(self._transferred_source)
        # reconstruction
        self.reconstructed_targets_batch = self._translate_to_vocabulary(self._teacher_forced_target)

    @staticmethod
    def _increase_if(counter, condition):
        return tf.cond(
            pred=condition,
            true_fn=lambda: counter.update,
            false_fn=lambda: counter.count
        )

    def _init_discriminator(self):
        if self.config['model']['discriminator_type'] == 'embedding':
            dense_inputs = self.config['discriminator_embedding']['encoder_hidden_states'][-1]
            if self.config['discriminator_embedding']['include_content_vector']:
                dense_inputs += self.config['model']['encoder_hidden_states'][-1]
            return EmbeddingDiscriminator(self.config['discriminator_embedding']['encoder_hidden_states'],
                                          dense_inputs,
                                          self.config['discriminator_embedding']['hidden_states'],
                                          self.discriminator_dropout_placeholder,
                                          self.config['discriminator_embedding']['bidirectional'],
                                          self.config['model']['cell_type'])
        if self.config['model']['discriminator_type'] == 'content':
            return ContentDiscriminator(self.config['model']['encoder_hidden_states'][-1],
                                        self.config['discriminator_content']['hidden_states'],
                                        self.discriminator_dropout_placeholder)

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
        encoded = self.encoder.encode_inputs_to_vector(embedding, input_lengths)
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

    def _translate_to_vocabulary(self, embeddings):
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
        random_words = self.config['margin_loss2']['random_words_size']
        padding_mask = tf.not_equal(self.target_batch, vocabulary_length)
        embedded_random_words = None
        if random_words > 0:
            input_shape = tf.shape(self.target_batch)
            embedded_random_words = self.embedding_container.get_random_words_embeddings(
                shape=(input_shape[0], input_shape[1], random_words)
            )
        return self.loss_handler.get_margin_loss_v2(self._target_embedding, self._teacher_forced_target,
                                                    embedded_random_words, padding_mask,
                                                    self.config['margin_loss2']['margin'])

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
            clip_val = self.config['wasserstein_loss']['clip_value']
            clip_discriminator = [p.assign(tf.clip_by_value(p, -clip_val, clip_val)) for p in discriminator_var_list]
            with tf.control_dependencies([discriminator_train_step]):
                return tf.group(*clip_discriminator)

    def _get_generator_train_step(self):
        with tf.variable_scope('TrainGeneratorSteps'):
            generator_optimizer = self._get_optimizer()
            generator_var_list = self.encoder.get_trainable_parameters() + self.decoder.get_trainable_parameters() + \
                                 self.embedding_container.get_trainable_parameters()
            generator_grads_and_vars = generator_optimizer.compute_gradients(
                self.generator_loss,
                colocate_gradients_with_ops=True,
                var_list=generator_var_list
            )
            return generator_optimizer.apply_gradients(generator_grads_and_vars)

    def _create_ratio_summary(self, nominator, denominator):
        return tf.cond(
            pred=tf.greater(denominator, 0),
            true_fn=lambda: tf.cast(nominator, tf.float32) / tf.cast(denominator, tf.float32),
            false_fn=lambda: 0.0
        )

    def _create_summaries(self):
        epoch_summary = tf.summary.scalar('epoch', self.epoch_counter.count)
        # train_generator_summary = tf.summary.scalar('train_generator', tf.cast(self.train_generator, dtype=tf.int8)),
        train_generator_summary = tf.summary.scalar(
            'train_generator',
            self._create_ratio_summary(self._generator_steps_counter.count, self._total_steps_counter.count)
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
            tf.summary.scalar('generator_loss', self.generator_loss),
            # tf.summary.scalar('apply_discriminator_loss_for_generator', tf.cast(
            #     self._apply_discriminator_loss_for_generator, tf.int8)),
            tf.summary.scalar(
                'apply_discriminator_loss_for_generator',
                self._create_ratio_summary(self._apply_discriminator_loss_for_generator_counter.count,
                                           self._generator_steps_counter.count)
            )
        ])
        return discriminator_step_summaries, generator_step_summaries


