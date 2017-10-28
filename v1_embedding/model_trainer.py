import tensorflow as tf
from v1_embedding.saver_wrapper import SaverWrapper
import yaml
import datetime
import os
from datasets.multi_batch_iterator import MultiBatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.gan_model import GanModel
from collections import Counter
from v1_embedding.logger import init_logger
from v1_embedding.pre_trained_embedding_handler import PreTrainedEmbeddingHandler
from datasets.classify_sentiment import classify


class ModelTrainer:
    def __init__(self, config_file, operational_config_file):
        self.config = config_file
        self.operational_config = operational_config_file
        self.train_statistics_frequency = self.operational_config['train_statistics_frequency']

        self.work_dir = None
        self.dataset_cache_dir = None
        self.embedding_dir = None
        self.summaries_dir = None

        self.saver_wrapper = None

        self.dataset_neg = YelpSentences(positive=False,
                                         limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(),
                                         dataset_name='neg')
        self.dataset_pos = YelpSentences(positive=True,
                                         limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(),
                                         dataset_name='pos')
        datasets = [self.dataset_neg, self.dataset_pos]
        self.embedding_handler = PreTrainedEmbeddingHandler(
            self.get_embedding_dir(),
            datasets,
            self.config['embedding']['word_size'],
            self.config['embedding']['min_word_occurrences']
        )

        contents = MultiBatchIterator.preprocess(datasets)
        # iterators
        self.batch_iterator = MultiBatchIterator(contents,
                                                 self.embedding_handler,
                                                 self.config['sentence']['min_length'],
                                                 self.config['trainer']['batch_size'])

        # set the model
        self.model = GanModel(self.config, self.operational_config, self.embedding_handler)

    def get_work_dir(self):
        if self.work_dir is None:
            self.work_dir = os.path.join(os.getcwd(), 'models', self.get_trainer_name())
        return self.work_dir

    def get_dataset_cache_dir(self):
        if self.dataset_cache_dir is None:
            self.dataset_cache_dir = os.path.join(self.get_work_dir(), 'dataset_cache')
        return self.dataset_cache_dir

    def get_embedding_dir(self):
        if self.embedding_dir is None:
            self.embedding_dir = os.path.join(self.get_work_dir(), 'embedding')
        return self.embedding_dir

    def get_summaries_dir(self):
        if self.summaries_dir is None:
            self.summaries_dir = os.path.join(self.get_work_dir(), 'tensorboard')
        return self.summaries_dir

    def get_trainer_name(self):
        return '{}_{}'.format(self.__class__.__name__, self.config['model']['discriminator_type'])

    def transfer_batch(self, sess, batch):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.target_batch: batch[1].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.target_lengths: batch[1].lengths,
            self.model.dropout_placeholder: 0.0,
            self.model.discriminator_dropout_placeholder: 0.0,
        }
        transferred_result, reconstruction_result = sess.run(
            [self.model.transferred_source_batch, self.model.reconstructed_targets_batch], feed_dict
        )
        end_of_sentence_index = self.embedding_handler.word_to_index[self.embedding_handler.end_of_sentence_token]
        # original source without paddings:
        original_source = self.remove_by_length(batch[0].sentences, batch[0].lengths)
        # original target without paddings:
        original_target = self.remove_by_length(batch[1].sentences, batch[1].lengths)
        # only take the prefix before EOS:
        transferred = []
        for s in transferred_result:
            if end_of_sentence_index in s:
                transferred.append(s[:s.tolist().index(end_of_sentence_index) + 1])
            else:
                transferred.append(s)
        reconstructed = []
        for s in reconstruction_result:
            if end_of_sentence_index in s:
                reconstructed.append(s[:s.tolist().index(end_of_sentence_index) + 1])
            else:
                reconstructed.append(s)
        return self.translate_to_string(original_target), \
               self.translate_to_string(reconstructed), \
               self.translate_to_string(original_source), \
               self.translate_to_string(transferred)
        # if print_to_file:
        #     now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     log_file_name = os.path.join('logs', '{}-epoch-{}.log'.format(now, epoch_num))
        #     original_source_strings, transferred_strings = self.print_to_file(
        #         original_source,
        #         transferred,
        #         self.embedding_handler,
        #         log_file_name
        #     )
        # else:
        #     # print the reconstruction
        #     original_target_strings, reconstructed_strings = self.print_side_by_side(
        #         original_target,
        #         reconstructed,
        #         'original_target: ',
        #         'reconstructed: ',
        #         self.embedding_handler
        #     )
        #     # print the transfer
        #     original_source_strings, transferred_strings = self.print_side_by_side(
        #         original_source,
        #         transferred,
        #         'original_source: ',
        #         'transferred: ',
        #         self.embedding_handler
        #     )
        # #evaluate the transfer
        # evaluation_prediction, evaluation_confidence = classify([' '.join(s) for s in transferred_strings])
        # evaluation_accuracy = Counter(evaluation_prediction)['pos'] / float(len(evaluation_prediction))
        # average_evaluation_confidence = sum(evaluation_confidence) / float(len(evaluation_confidence))
        # if print_to_file:
        #     with open(os.path.join('logs', 'accuracy.log'), 'a+') as f:
        #         f.write('Date: {}, Epoch: {}, Acc: {}, Confidence: {}\n'.format(now, epoch_num, evaluation_accuracy,
        #                                                                         average_evaluation_confidence))
        # else:
        #     print('Transferred evaluation acc: {} with average confidence of: {}'.format(
        #         evaluation_accuracy, average_evaluation_confidence)
        #     )
        #
        # if return_result_as_summary:
        #     return sess.run(
        #         self.model.evaluation_summary,
        #         {
        #             self.model.text_watcher.placeholders['original_source']: [' '.join(s) for s in
        #                                                                       original_source_strings],
        #             self.model.text_watcher.placeholders['original_target']: [' '.join(s) for s in
        #                                                                       original_target_strings],
        #             self.model.text_watcher.placeholders['transferred']: [' '.join(s) for s in transferred_strings],
        #             self.model.text_watcher.placeholders['reconstructed']: [' '.join(s) for s in reconstructed_strings],
        #         })
        # else:
        #     return None

    def translate_to_string(self, indices):
        return [' '.join(s) for s in self.embedding_handler.get_index_to_word(indices)]

    def print_to_file(self, global_step, epoch_number, sentences, file_name):
        with open(file_name, 'a+') as f:
            f.write('global step: {} epoch: {}\n'.format(global_step, epoch_number))
            for i in range(len(sentences)):
                f.write('{}\n'.format(sentences[i]))
            f.write('-------------------------------------------------------------\n')

    @staticmethod
    def remove_by_length(sentences, lengths):
        return [[
            word for word_index, word in enumerate(sentence) if word_index < lengths[sentence_index]
        ] for sentence_index, sentence in enumerate(sentences)]

    def do_train_loop(self, name):
        self.saver_wrapper = SaverWrapper(self.get_work_dir(), self.get_trainer_name())
        session_config = tf.ConfigProto(log_device_placement=self.operational_config['print_device'],
                                        allow_soft_placement=True)
        session_config.gpu_options.allow_growth = False
        if self.operational_config['run_optimizer']:
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=session_config) as sess:
            use_tensorboard = self.operational_config['tensorboard_frequency'] > 0
            summary_writer_train = tf.summary.FileWriter(os.path.join(self.get_summaries_dir(), 'train'),
                                                         sess.graph) if use_tensorboard else None
            summary_writer_validation = tf.summary.FileWriter(
                os.path.join(self.get_summaries_dir(), 'validation')) if use_tensorboard else None

            sess.run(tf.global_variables_initializer())
            if self.operational_config['load_model']:
                self.saver_wrapper.load_model(sess)

            self.do_before_train_loop(sess)

            train_statistics_file = os.path.join('logs', '{}_train_stats.log'.format(name))
            with open(train_statistics_file, 'w') as train_statistics:
                line = ';'.join([
                    'time',
                    'global_step',
                    'epoch_num',
                    'batch_index',
                    'discriminator_accuracy',
                    'discriminator_loss',
                    'reconstruction_loss',
                    'generator_loss',
                    'generator_steps_to_total_steps',
                    'apply_discriminator_steps_to_generator_steps',
                ])
                train_statistics.write("{}\n".format(line))
                global_step = 0
                for epoch_num in range(self.config['trainer']['number_of_epochs']):
                    print('epoch {} of {}'.format(epoch_num + 1, self.config['trainer']['number_of_epochs']))
                    self.do_before_epoch(sess, global_step, epoch_num)
                    for batch_index, batch in enumerate(self.batch_iterator):
                        if (global_step % self.operational_config['validation_batch_frequency']) == 1:
                            validation_summaries = self.do_validation_batch(
                                sess, global_step, epoch_num, batch, use_tensorboard, name
                            )
                            if validation_summaries:
                                summary_writer_validation.add_summary(validation_summaries, global_step=global_step)
                        extract_summaries = use_tensorboard and \
                                            (global_step % self.operational_config['tensorboard_frequency'] == 1)
                        train_summaries = self.do_train_batch(sess, global_step, epoch_num, batch_index, batch,
                                                              train_statistics, extract_summaries=extract_summaries)
                        if train_summaries:
                            summary_writer_train.add_summary(train_summaries, global_step=global_step)
                        global_step += 1
                    self.do_after_epoch(sess, global_step, epoch_num)
            self.do_after_train_loop(sess)

    def do_before_train_loop(self, sess):
        sess.run(self.model.embedding_container.assign_embedding(), {
            self.model.embedding_container.embedding_placeholder: self.embedding_handler.embedding_np
        })

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch, train_statistics_file,
                       extract_summaries):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.target_batch: batch[1].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.target_lengths: batch[1].lengths,
            self.model.dropout_placeholder: self.config['model']['dropout'],
            self.model.discriminator_dropout_placeholder: self.config['model']['discriminator_dropout'],
        }
        summary = None
        # write to statistics file
        if (global_step % self.train_statistics_frequency) == 1:
            execution_list = [
                self.model.master_step,
                self.model.discriminator_loss,
                self.model.reconstruction_loss,
                self.model.generator_loss,
                self.model.accuracy,
                self.model.total_steps_counter.count,
                self.model.generator_steps_counter.count,
                self.model.apply_discriminator_loss_for_generator_counter.count,
                self.model.summary_step
            ]
            if extract_summaries:
                _, discriminator_loss, reconstruction_loss, generator_loss, accuracy, total_steps, generator_steps, apply_discriminator_loss_for_generator_counter, summary = sess.run(
                    execution_list, feed_dict)
            else:
                _, discriminator_loss, reconstruction_loss, generator_loss, accuracy, total_steps, generator_steps, apply_discriminator_loss_for_generator_counter = sess.run(
                    execution_list[:-1], feed_dict)

            line = ';'.join([
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                str(global_step),
                str(epoch_num),
                str(batch_index),
                str(accuracy),
                str(discriminator_loss),
                str(reconstruction_loss),
                str(generator_loss),
                str(float(generator_steps) / float(total_steps)),
                str(float(apply_discriminator_loss_for_generator_counter) / float(generator_steps))
            ])
            train_statistics_file.write("{}\n".format(line))
        else:
            if extract_summaries:
                _, summary = sess.run([self.model.master_step, self.model.summary_step], feed_dict)
            else:
                _ = sess.run(self.model.master_step, feed_dict)
        return summary

    def do_validation_batch(self, sess, global_step, epoch_num, batch, extract_summary, name):
        target, reconstructed, source, transferred = self.transfer_batch(sess, batch)
        self.print_to_file(global_step, epoch_num, source, os.path.join('logs', '{}_source.log'.format(name)))
        self.print_to_file(global_step, epoch_num, target, os.path.join('logs', '{}_target.log'.format(name)))
        self.print_to_file(global_step, epoch_num, transferred,
                           os.path.join('logs', '{}_transferred.log'.format(name)))
        self.print_to_file(global_step, epoch_num, reconstructed,
                           os.path.join('logs', '{}_reconstructed.log'.format(name)))
        return sess.run(
                self.model.evaluation_summary,
                {
                    self.model.text_watcher.placeholders['original_source']: source,
                    self.model.text_watcher.placeholders['original_target']: target,
                    self.model.text_watcher.placeholders['transferred']: transferred,
                    self.model.text_watcher.placeholders['reconstructed']: reconstructed,
                }) if extract_summary else None

    def do_after_train_loop(self, sess):
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        print('model loaded, sample sentences:')
        for batch in self.batch_iterator:
            original_target, reconstructed, original_source, transferred = self.transfer_batch(sess, batch)
            for i in range(len(original_target)):
                print('original_target: {}'.format(original_target[i]))
                print('reconstructed: {}'.format(reconstructed[i]))
                print('original_source: {}'.format(original_source[i]))
                print('transferred: {}'.format(transferred[i]))
            break

    def do_before_epoch(self, sess, global_step, epoch_num):
        sess.run(self.model.epoch_counter.update)

    def do_after_epoch(self, sess, global_step, epoch_num):
        if epoch_num % 10 == 0:
            # activate the saver
            self.saver_wrapper.save_model(sess, global_step=global_step)

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("config/gan.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)
    init_logger(name)
    print('------------ Config ------------')
    print(yaml.dump(config))
    print('------------ Operational Config ------------')
    print(yaml.dump(operational_config))
    ModelTrainer(config, operational_config).do_train_loop(name)
