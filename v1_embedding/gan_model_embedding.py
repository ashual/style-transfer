import tensorflow as tf

from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
from v1_embedding.gan_model import GanModel

# in this model the discriminator differentiates between the true target from transferred targets using professor
# forcing on the decoded outputs of teh decoder
class GanModelEmbedding(GanModel):
    def __init__(self, config_file, operational_config_file, embedding_handler):
        GanModel.__init__(self, config_file, operational_config_file, embedding_handler)

        self.discriminator = EmbeddingDiscriminator(self.config['discriminator_embedding']['hidden_states'],
                                                    self.config['discriminator_embedding']['dense_hidden_size'],
                                                    self.discriminator_dropout_placeholder,
                                                    self.config['discriminator_embedding']['bidirectional'])

        # losses and accuracy:
        self.discriminator_step_prediction, self.discriminator_loss_on_discriminator_step,\
        self.discriminator_accuracy_for_discriminator = self.get_discriminator_loss(
            self.source_batch, self.source_lengths, self.target_batch, self.target_lengths
        )

        self.generator_step_prediction, self.discriminator_accuracy_for_generator, \
        self.discriminator_loss_on_generator_step, self.reconstruction_loss_on_generator_step, \
        self.content_vector_loss_on_generator_step = self.get_generator_loss(
            self.source_batch, self.source_lengths, self.target_batch, self.target_lengths
        )
        self.generator_loss = self.config['model']['reconstruction_coefficient'] * \
                              self.reconstruction_loss_on_generator_step \
                              + self.config['model']['semantic_distance_coefficient'] * \
                                self.content_vector_loss_on_generator_step - self.discriminator_loss_on_generator_step

        # train steps
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('TrainSteps'):
            with tf.variable_scope('TrainDiscriminatorSteps'):
                discriminator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
                discriminator_var_list = self.discriminator.get_trainable_parameters()
                discriminator_grads_and_vars = discriminator_optimizer.compute_gradients(
                    self.discriminator_loss_on_discriminator_step,
                    colocate_gradients_with_ops=True,
                    var_list=discriminator_var_list
                )
                with tf.control_dependencies(update_ops):
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
                with tf.control_dependencies(update_ops):
                    self.generator_train_step = generator_optimizer.apply_gradients(generator_grads_and_vars)

    def _teacher_force_target(self, target_batch, target_lengths):
        encoded_target = self._encode(target_batch, target_lengths)
        target_embedding = self.embedding_container.embed_inputs(target_batch)
        return self.decoder.do_teacher_forcing(
            encoded_target, target_embedding[:, :-1, :], target_lengths, domain_identifier=None
        )

    def _get_discriminator_prediction_loss_and_accuracy(self, transferred_source, teacher_forced_target):
        sentence_length = tf.shape(teacher_forced_target)[1]
        transferred_source_normalized = transferred_source[:, :sentence_length, :]
        prediction = self.discriminator.predict(tf.concat((transferred_source_normalized, teacher_forced_target),
                                                          axis=0))
        transferred_batch_size = tf.shape(transferred_source)[0]

        prediction_transferred = prediction[:transferred_batch_size, :]
        prediction_target = prediction[transferred_batch_size:, :]
        total_loss, total_accuracy = self.loss_handler.get_discriminator_loss(prediction_transferred, prediction_target)

        return prediction, total_loss, total_accuracy

    def get_discriminator_loss(self, source_batch, source_lengths, target_batch, target_lengths):
        # calculate the source-encoded-as-target loss
        sentence_length = tf.shape(source_batch)[1]
        transferred_source = self._transfer(source_batch, source_lengths)[:, :sentence_length, :]

        # calculate the teacher forced loss
        teacher_forced_target = self._teacher_force_target(target_batch, target_lengths)

        return self._get_discriminator_prediction_loss_and_accuracy(transferred_source, teacher_forced_target)

    def get_generator_loss(self, source_batch, source_lengths, target_batch, target_lengths):
        encoded_source = self._encode(source_batch, source_lengths)

        # reconstruction loss - recover target
        teacher_forced_target = self._teacher_force_target(target_batch, target_lengths)
        reconstruction_loss = self._reconstruction_loss(target_batch, teacher_forced_target)

        # semantic vector distance
        transferred_source = self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)
        encoded_again = self.encoder.encode_inputs_to_vector(transferred_source, None, domain_identifier=None)
        semantic_distance_loss = self.loss_handler.get_context_vector_distance_loss(encoded_source, encoded_again)

        # professor forcing loss source
        discriminator_prediction, discriminator_loss, discriminator_accuracy = \
            self._get_discriminator_prediction_loss_and_accuracy(
                transferred_source, teacher_forced_target
            )

        return discriminator_prediction, discriminator_accuracy, discriminator_loss, reconstruction_loss, \
               semantic_distance_loss