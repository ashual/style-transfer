import yaml
import tensorflow as tf
import time
from datasets.batch_iterator import BatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.loss_handler import LossHandler
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.model_trainer_base import ModelTrainerBase


class ModelTrainerValidation(ModelTrainerBase):
    def __init__(self, config_file, operational_config_file):
        ModelTrainerBase.__init__(self, config_file=config_file, operational_config_file=operational_config_file)

        self.best_validation_acc = tf.Variable(-1.0, trainable=False)

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        # placeholder for sentences (batch, time)=> index of word s.t the padding is on the left
        self.left_padded_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for sentences (batch, time)=> index of word s.t the padding is on the right
        self.right_padded_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.dataset = YelpSentences(positive=False, limit_sentences=self.config['sentence']['limit'],
                                     dataset_cache_dir=self.get_dataset_cache_dir())
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.get_embedding_dir(),
            [self.dataset],
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

        self.loss_handler = LossHandler(self.embedding_handler.get_vocabulary_length())

        # "One Hot Vector" -> Embedded Vector (w2v)
        left_padded_embeddings = self.embedding_translator.embed_inputs(self.left_padded_batch)
        right_padded_embedding = self.embedding_translator.embed_inputs(self.right_padded_batch)
        # Embedded Vector (w2v) -> Encoded (constant length)
        encoded = self.encoder.encode_inputs_to_vector(left_padded_embeddings, self.source_identifier)
        # Encoded -> Decoded
        decoded = self.decoder.do_teacher_forcing(encoded, right_padded_embedding[:, :-1, :], self.source_identifier)
        # decoded -> logits
        logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(decoded)
        # cross entropy loss
        self.loss = self.loss_handler.get_sentence_reconstruction_loss(self.right_padded_batch, logits)
        # training
        optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
        self.train_step = optimizer.apply_gradients(grads_and_vars)
        # maximal word for each step
        self.outputs = self.embedding_translator.translate_logits_to_words(logits)
        # accuracy
        self.accuracy = self.loss_handler.get_accuracy(self.right_padded_batch, self.outputs)
        self.best_loss = float('inf')
        self.loss_output = float('inf')
        self.epoch = tf.Variable(0, trainable=False)
        self.sentence_length = tf.Variable(self.config['sentence']['min_length'], trainable=False)
        # summaries
        loss_summary = tf.summary.scalar('loss', self.loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        epoch = tf.summary.scalar('epoch', self.epoch)
        sentence_length = tf.summary.scalar('sentence_length', self.sentence_length)
        weight_summaries = tf.summary.merge(self.embedding_translator.get_trainable_parameters_summaries() +
                                            self.encoder.get_trainable_parameters_summaries() +
                                            self.decoder.get_trainable_parameters_summaries())
        gradient_summaries = tf.summary.merge([item for sublist in
                                               [BaseModel.create_summaries(g) for g, v in grads_and_vars]
                                               for item in sublist])
        gradient_global_norm = tf.summary.scalar('gradient_global_norm', tf.global_norm([g for g, v in grads_and_vars]))
        # needed by ModelTrainerBase
        self.batch_iterator = BatchIterator(self.dataset, self.embedding_handler,
                                            sentence_len=self.config['sentence']['min_length'],
                                            batch_size=self.config['model']['batch_size'])

        self.batch_iterator_validation = BatchIterator(self.dataset,
                                                       self.embedding_handler,
                                                       sentence_len=self.config['sentence']['min_length'],
                                                       batch_size=1000)

        self.train_summaries = tf.summary.merge([loss_summary, accuracy_summary, weight_summaries, gradient_summaries,
                                                 gradient_global_norm, epoch, sentence_length])
        self.validation_summaries = tf.summary.merge([accuracy_summary, weight_summaries, loss_summary, epoch,
                                                      sentence_length])

    def do_before_train_loop(self, sess):
        best_validation_acc = sess.run(self.best_validation_acc)
        print('starting validation accuracy: {}'.format(best_validation_acc))
        sess.run(self.embedding_translator.assign_embedding(), {
            self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
        })

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch):
        sess.run([tf.assign(self.epoch, epoch_num)])
        feed_dict = {
            self.left_padded_batch: batch.left_padded_sentences,
            self.right_padded_batch: batch.right_padded_sentences,
            self.dropout_placeholder: self.config['model']['dropout'],
            self.encoder.should_print: self.operational_config['debug'],
            self.decoder.should_print: self.operational_config['debug'],
        }
        train_summaries = None
        execution_list = [self.train_step, self.loss, self.outputs, self.accuracy, self.train_summaries]

        # print results
        if batch_index % 100 == 0:
            start_time = time.time()
            _, loss_output, decoded_output, batch_acc, train_summaries = sess.run(execution_list, feed_dict)
            total_time = time.time() - start_time
            self.print_side_by_side(
                self.remove_by_mask(batch.right_padded_sentences, batch.right_padded_masks),
                self.remove_by_mask(decoded_output, batch.right_padded_masks),
                'original: ',
                'reconstructed: ',
                self.embedding_handler
            )
            print('epoch-index: {} batch-index: {} acc: {} loss: {} runtime: {}'.format(epoch_num, batch_index,
                                                                                        batch_acc, loss_output,
                                                                                        total_time))
            print()
        else:
            # will not run summaries
            _, loss_output, decoded_output, batch_acc = sess.run(execution_list[:-1], feed_dict)

        return train_summaries

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, batch):
        feed_dict = {
            self.left_padded_batch: batch.left_padded_sentences,
            self.right_padded_batch: batch.right_padded_sentences,
            self.dropout_placeholder: 0.0,
            self.encoder.should_print: self.operational_config['debug'],
            self.decoder.should_print: self.operational_config['debug'],
        }
        self.loss_output, validation_acc, validation_summaries, best_validation_acc = sess.run(
            [self.loss,
             self.accuracy,
             self.validation_summaries,
             self.best_validation_acc],
            feed_dict)
        print('validation: loss - {}, accuracy - {} (best - {})'.format(
            self.loss_output, validation_acc, best_validation_acc)
        )
        if validation_acc > best_validation_acc:
            print('saving model, former best accuracy {} current best accuracy {}\n'.
                  format(best_validation_acc, validation_acc))
            if self.saver_wrapper.save_model(sess, global_step=global_step):
                sess.run([tf.assign(self.best_validation_acc, validation_acc)])

        return validation_summaries

    def do_after_train_loop(self, sess):
        best_validation_acc = sess.run(self.best_validation_acc)
        print('best validation accuracy: {}'.format(best_validation_acc))
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        for batch in self.batch_iterator_validation:
            feed_dict = {
                self.left_padded_batch: batch.left_padded_sentences,
                self.right_padded_batch: batch.right_padded_sentences,
                self.dropout_placeholder: 0.0,
                self.encoder.should_print: self.operational_config['debug'],
                self.decoder.should_print: self.operational_config['debug'],
            }
            validation_acc = sess.run(self.accuracy, feed_dict)
            print('tested validation accuracy: {}'.format(validation_acc))
            print()
            break

    def do_before_epoch(self, sess):
        enlarge = False
        message = ''
        if not self.config['model']['curriculum_training'] or\
                        self.batch_iterator.sentence_len >= self.config['sentence']['max_length']:
            return
        # The loss is ok, but keep decreasing
        if config['loss']['upper_range'] >= self.loss_output > config['loss']['lower_range'] and \
                        self.loss_output >= self.best_loss:
            enlarge = True
            message = "loss is in range {}>={}>{} and doesn't decrease - best:{}, current: {}".format(
                config['loss']['upper_range'],
                self.loss_output,
                config['loss']['lower_range'],
                self.best_loss,
                self.loss_output
            )
        elif config['loss']['lower_range'] >= self.loss_output:
            enlarge = True
            message = "loss ({}) is smaller than lower_range: {}".format(self.loss_output,
                                                                         config['loss']['lower_range'])

        if enlarge:
            num_of_words = self.batch_iterator.sentence_len + 1
            sess.run([tf.assign(self.sentence_length, num_of_words)])
            print('Moving from {} to {} words because {}'.format(num_of_words - 1, num_of_words, message))
            self.batch_iterator.sentence_len = num_of_words
            self.batch_iterator_validation.sentence_len = num_of_words
            self.best_loss = float('inf')
        else:
            self.best_loss = min(self.best_loss, self.loss_output)

    def do_after_epoch(self, sess):
        pass


if __name__ == "__main__":
    with open("config/validation_word_index.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)

    ModelTrainerValidation(config, operational_config).do_train_loop()
