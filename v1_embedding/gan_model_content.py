import tensorflow as tf

from v1_embedding.content_discriminator import ContentDiscriminator
from v1_embedding.gan_model import GanModel

# in this model the discriminator differentiates between encoded source and encoded targets
class GanModelContent(GanModel):
    def __init__(self, config_file, operational_config_file, embedding_handler):
        GanModel.__init__(self, config_file, operational_config_file, embedding_handler)

        self.discriminator = ContentDiscriminator(self.config['model']['encoder_hidden_states'][-1],
                                                  self.config['discriminator_content']['hidden_states'],
                                                  self.discriminator_dropout_placeholder)

        # losses and accuracy:
        self.discriminator_step_prediction, self.discriminator_loss, self.discriminator_accuracy_for_discriminator = \
            self.get_discriminator_loss(
                self.left_padded_source_batch,
                self.left_padded_target_batch
            )

        self.generator_step_prediction, self.generator_loss, self.discriminator_accuracy_for_generator, \
        self.discriminator_loss_on_generator_step = \
            self.get_generator_loss(
                self.left_padded_source_batch,
                self.left_padded_target_batch,
                self.right_padded_target_batch
            )

        # train steps
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('TrainSteps'):
            with tf.variable_scope('TrainDiscriminatorSteps'):
                discriminator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
                discriminator_var_list = self.discriminator.get_trainable_parameters()
                discriminator_grads_and_vars = discriminator_optimizer.compute_gradients(
                    self.discriminator_loss,
                    colocate_gradients_with_ops=True,
                    var_list=discriminator_var_list
                )
                with tf.control_dependencies(update_ops):
                    self.discriminator_train_step = discriminator_optimizer.apply_gradients(
                        discriminator_grads_and_vars)
            with tf.variable_scope('TrainGeneratorSteps'):
                generator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
                generator_var_list = self.encoder.get_trainable_parameters() + \
                                     self.decoder.get_trainable_parameters() + \
                                     self.embedding_translator.get_trainable_parameters()
                generator_grads_and_vars = generator_optimizer.compute_gradients(
                    self.generator_loss,
                    colocate_gradients_with_ops=True,
                    var_list=generator_var_list
                )
                with tf.control_dependencies(update_ops):
                    self.generator_train_step = generator_optimizer.apply_gradients(generator_grads_and_vars)

    def _get_discriminator_prediction_loss_and_accuracy(self, encoded_source, encoded_target):
        prediction = self.discriminator.predict(tf.concat((encoded_source, encoded_target),
                                                          axis=0))
        transferred_batch_size = tf.shape(encoded_source)[0]

        prediction_transferred = prediction[:transferred_batch_size, :]
        prediction_target = prediction[transferred_batch_size:, :]
        total_loss, total_accuracy = self.loss_handler.get_discriminator_loss(prediction_transferred, prediction_target)

        return prediction, total_loss, total_accuracy

    def get_discriminator_loss(self, left_padded_source_batch, left_padded_target_batch):
        encoded_source = self._encode(left_padded_source_batch)
        encoded_target = self._encode(left_padded_target_batch)
        return self._get_discriminator_prediction_loss_and_accuracy(encoded_source, encoded_target)

    def get_generator_loss(self, left_padded_source_batch, left_padded_target_batch, right_padded_target_batch):
        encoded_source = self._encode(left_padded_source_batch)

        # reconstruction loss - recover target
        encoded_target = self._encode(left_padded_target_batch)
        right_padded_target_embedding = self.embedding_translator.embed_inputs(right_padded_target_batch)
        teacher_forced_target = self.decoder.do_teacher_forcing(encoded_target,
                                                                right_padded_target_embedding[:, :-1, :],
                                                                domain_identifier=None)
        reconstructed_target_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
            teacher_forced_target)
        reconstruction_loss = self.loss_handler.get_sentence_reconstruction_loss(right_padded_target_batch,
                                                                                 reconstructed_target_logits)

        # semantic vector distance
        transferred_source = self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)
        encoded_again = self.encoder.encode_inputs_to_vector(transferred_source, domain_identifier=None)
        semantic_distance_loss = self.loss_handler.get_context_vector_distance_loss(encoded_source, encoded_again)

        # professor forcing loss source
        discriminator_prediction, discriminator_loss, discriminator_accuracy = \
            self._get_discriminator_prediction_loss_and_accuracy(
                encoded_source, encoded_target
            )

        total_loss = self.config['model']['reconstruction_coefficient'] * reconstruction_loss \
                     + self.config['model']['semantic_distance_coefficient'] * semantic_distance_loss \
                     - discriminator_loss
        return discriminator_prediction, total_loss, discriminator_accuracy, discriminator_loss