import tensorflow as tf
import time
import yaml
from datasets.multi_batch_iterator import MultiBatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
from v1_embedding.loss_handler import LossHandler
from v1_embedding.model_trainer_base import ModelTrainerBase
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler


# this model tries to transfer from one domain to another.
# 1. the encoder doesn't know the domain it is working on
# 2. target are encoded and decoded (to target) then cross entropy loss is applied between the origin and the result
# 3. source is encoded decoded to target and encoded again, then L2 loss is applied between the context vectors.
# 4. an adversarial component is trained to distinguish true target from transferred targets using professor forcing
class ModelTrainer(ModelTrainerBase):
    def __init__(self, config_file, operational_config_file):
        ModelTrainerBase.__init__(self, config_file=config_file, operational_config_file=operational_config_file)

        self.epsilon = 0.001

        # placeholders for dropouts
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        self.discriminator_dropout_placeholder = tf.placeholder(tf.float32, shape=())
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the left
        self.left_padded_source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word s.t the padding is on the right
        self.right_padded_source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the left
        self.left_padded_target_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for target sentences (batch, time)=> index of word s.t the padding is on the right
        self.right_padded_target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.dataset_neg = YelpSentences(positive=False, limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.dataset_cache_dir, dataset_name='neg')
        self.dataset_pos = YelpSentences(positive=True, limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.dataset_cache_dir, dataset_name='pos')
        datasets = [self.dataset_neg, self.dataset_pos]
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.embedding_dir,
            datasets,
            self.config['embedding']['word_size'],
            self.config['embedding']['min_word_occurrences']
        )
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
        self.discriminator = EmbeddingDiscriminator(self.config['model']['discriminator_hidden_states'],
                                                    self.config['model']['discriminator_dense_hidden_size'],
                                                    self.discriminator_dropout_placeholder,
                                                    self.config['model']['bidirectional_discriminator'])
        self.loss_handler = LossHandler(self.embedding_handler.get_vocabulary_length())

        # losses:
        self.discriminator_loss, self.discriminator_accuracy_for_discriminator = self.get_discriminator_loss(
            self.left_padded_source_batch,
            self.left_padded_target_batch,
            self.right_padded_target_batch
        )

        self.generator_loss, self.discriminator_accuracy_for_generator = self.get_generator_loss(
            self.left_padded_source_batch,
            self.left_padded_target_batch,
            self.right_padded_target_batch
        )

        # train steps
        discriminator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
        discriminator_var_list = self.discriminator.get_trainable_parameters()
        discriminator_grads_and_vars = discriminator_optimizer.compute_gradients(
            self.discriminator_loss,
            colocate_gradients_with_ops=True,
            var_list=discriminator_var_list
        )
        self.discriminator_train_step = discriminator_optimizer.apply_gradients(discriminator_grads_and_vars)

        generator_optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
        generator_var_list = self.encoder.get_trainable_parameters() + \
                             self.decoder.get_trainable_parameters() + \
                             self.embedding_translator.get_trainable_parameters()
        generator_grads_and_vars = generator_optimizer.compute_gradients(
            self.generator_loss,
            colocate_gradients_with_ops=True,
            var_list=generator_var_list
        )
        self.generator_train_step = generator_optimizer.apply_gradients(generator_grads_and_vars)

        # do transfer
        transferred_embeddings = self._transfer(self.left_padded_source_batch)
        transferred_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(transferred_embeddings)
        self.transfer = self.embedding_translator.translate_logits_to_words(transferred_logits)

        # iterators
        self.batch_iterator = MultiBatchIterator(datasets,
                                                 self.embedding_handler,
                                                 self.config['sentence']['min_length'],
                                                 self.config['model']['batch_size'])

        # iterators
        self.batch_iterator_validation = MultiBatchIterator(datasets,
                                                            self.embedding_handler,
                                                            self.config['sentence']['min_length'],
                                                            2)
        # train loop parameters:
        self.history_weight = 0.95
        self.running_acc = None
        self.train_generator = None

    def _stable_log(self, input):
        input = tf.maximum(input, self.epsilon)
        input = tf.minimum(input, 1.0 - self.epsilon)
        return tf.log(input)

    def _encode(self, left_padded_input):
        embedding = self.embedding_translator.embed_inputs(left_padded_input)
        return self.encoder.encode_inputs_to_vector(embedding, domain_identifier=None)

    def _transfer(self, left_padded_source):
        encoded_source = self._encode(left_padded_source)
        return self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)

    def _teacher_force_target(self, left_padded_target_batch, right_padded_target_batch):
        encoded_target = self._encode(left_padded_target_batch)
        right_padded_target_embedding = self.embedding_translator.embed_inputs(right_padded_target_batch)
        return self.decoder.do_teacher_forcing(encoded_target,
                                               right_padded_target_embedding[:, :-1, :],
                                               domain_identifier=None)

    def _get_discriminator_prediction_loss_and_accuracy(self, transferred_source, teacher_forced_target):
        discriminator_prediction_fake_target = self.discriminator.predict(transferred_source)
        transferred_accuracy = tf.reduce_mean(tf.cast(tf.less(discriminator_prediction_fake_target, 0.5), tf.float32))
        transferred_loss = -tf.reduce_mean(self._stable_log(1.0 - discriminator_prediction_fake_target))

        discriminator_prediction_target = self.discriminator.predict(teacher_forced_target)
        target_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(discriminator_prediction_target, 0.5), tf.float32))
        target_loss = -tf.reduce_mean(self._stable_log(discriminator_prediction_target))

        # total loss is the sum of losses
        total_loss = transferred_loss + target_loss
        # total accuracy is the avg of accuracies
        total_accuracy = 0.5 * (transferred_accuracy + target_accuracy)
        return total_loss, total_accuracy

    def get_discriminator_loss(self, left_padded_source_batch, left_padded_target_batch, right_padded_target_batch):
        # calculate the source-encoded-as-target loss
        sentence_length = tf.shape(left_padded_source_batch)[1]
        transferred_source = self._transfer(left_padded_source_batch)[:, :sentence_length, :]

        # calculate the teacher forced loss
        teacher_forced_target = self._teacher_force_target(left_padded_target_batch, right_padded_target_batch)

        return self._get_discriminator_prediction_loss_and_accuracy(transferred_source, teacher_forced_target)

    def get_generator_loss(self, left_padded_source_batch, left_padded_target_batch, right_padded_target_batch):
        encoded_source = self._encode(left_padded_source_batch)

        # reconstruction loss - recover target
        teacher_forced_target = self._teacher_force_target(left_padded_target_batch, right_padded_target_batch)
        reconstructed_target_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
            teacher_forced_target)
        reconstruction_loss = self.loss_handler.get_sentence_reconstruction_loss(right_padded_target_batch,
                                                                                 reconstructed_target_logits)

        # semantic vector distance
        transferred_source = self.decoder.do_iterative_decoding(encoded_source, domain_identifier=None)
        encoded_again = self.encoder.encode_inputs_to_vector(transferred_source, domain_identifier=None)
        semantic_distance_loss = self.loss_handler.get_context_vector_distance_loss(encoded_source, encoded_again)

        # professor forcing loss source
        discriminator_loss, discriminator_accuracy = self._get_discriminator_prediction_loss_and_accuracy(
            transferred_source, teacher_forced_target
        )

        total_loss = self.config['model']['reconstruction_coefficient'] * reconstruction_loss \
                     + self.config['model']['semantic_distance_coefficient'] * semantic_distance_loss \
                     - discriminator_loss
        return total_loss, discriminator_accuracy

    def before_train_generator(self):
        self.train_generator = True
        self.running_acc = 1.0

    def before_train_discriminator(self):
        self.train_generator = False
        self.running_acc = 0.5

    def update_running_accuracy(self, new_accuracy_score):
        return self.history_weight * self.running_acc + (1-self.history_weight) * new_accuracy_score

    def do_generator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary):
        # TODO: outputs to measure progress, summaries
        print('started generator')
        print('running acc: {}'.format(self.running_acc))
        execution_list = [self.generator_loss, self.discriminator_accuracy_for_generator]
        loss, acc = sess.run(execution_list, feed_dictionary)
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        if self.running_acc >= acc and self.running_acc >= 0.5 + self.epsilon:
            # the generator is still improving
            self.running_acc = self.update_running_accuracy(acc)
            print('new running acc: {}'.format(self.running_acc))
            sess.run(self.generator_train_step, feed_dictionary)
        else:
            print('generator too good - training discriminator')
            # the generator is no longer improving, will train discriminator next
            self.before_train_discriminator()
            self.do_discriminator_train(sess, global_step, epoch_num, batch_index, feed_dictionary)

    def do_discriminator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary):
        # TODO: outputs to measure progress, summaries
        print('started discriminator')
        print('running acc: {}'.format(self.running_acc))
        execution_list = [self.discriminator_loss, self.discriminator_accuracy_for_discriminator]
        loss, acc = sess.run(execution_list, feed_dictionary)
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        if self.running_acc <= acc and self.running_acc <= 1.0 - self.epsilon:
            # the discriminator is still improving
            self.running_acc = self.update_running_accuracy(acc)
            print('new running acc: {}'.format(self.running_acc))
            sess.run(self.discriminator_train_step, feed_dictionary)
        else:
            print('discriminator too good - training generator')
            # the discriminator is no longer improving, will train generator next
            self.before_train_generator()
            self.do_generator_train(sess, global_step, epoch_num, batch_index, feed_dictionary)

    def do_before_train_loop(self, sess):
        sess.run(self.embedding_translator.assign_embedding(), {
            self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
        })
        self.before_train_discriminator()

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch):
        feed_dict = {
            self.left_padded_source_batch: batch[0].left_padded_sentences,
            self.left_padded_target_batch: batch[1].left_padded_sentences,
            self.right_padded_source_batch: batch[0].right_padded_sentences,
            self.right_padded_target_batch: batch[1].right_padded_sentences,
            self.dropout_placeholder: self.config['model']['dropout'],
            self.discriminator_dropout_placeholder: self.config['model']['discriminator_dropout'],
            self.encoder.should_print: self.operational_config['debug'],
            self.decoder.should_print: self.operational_config['debug'],
            self.discriminator.should_print: self.operational_config['debug'],
            self.embedding_translator.should_print: self.operational_config['debug'],
        }
        print('batch len: {}'.format(batch[0].get_len()))
        if self.train_generator:
            # should train the generator
            return self.do_generator_train(sess, global_step, epoch_num, batch_index, feed_dict)
        else:
            # should train discriminator
            return self.do_discriminator_train(sess, global_step, epoch_num, batch_index, feed_dict)

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, batch):
        feed_dict = {
            self.left_padded_source_batch: batch[0].left_padded_sentences,
            self.dropout_placeholder: 0.0,
            self.encoder.should_print: self.operational_config['debug'],
            self.decoder.should_print: self.operational_config['debug'],
            self.embedding_translator.should_print: self.operational_config['debug'],
        }
        transferred_result = sess.run(self.transfer, feed_dict)
        end_of_sentence_index = self.embedding_handler.word_to_index[self.embedding_handler.end_of_sentence_token]
        # only take the prefix before EOS:
        transferred_result = [s[:s.tolist().index(end_of_sentence_index) + 1] for s in transferred_result if
                             end_of_sentence_index in s]
        # print the transfer
        # self.print_side_by_side(
        #     self.remove_by_mask(batch[0].right_padded_sentences, batch[0].right_padded_masks),
        #     transfered_result,
        #     'original: ',
        #     'transferred: ',
        #     self.embedding_handler
        # )
        # print the accuracy traces:

    def do_after_train_loop(self, sess):
        pass

    def do_before_epoch(self, sess):
        pass

    def do_after_epoch(self, sess):
        pass


if __name__ == "__main__":
    with open("config/gan.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)

    ModelTrainer(config, operational_config).do_train_loop()
