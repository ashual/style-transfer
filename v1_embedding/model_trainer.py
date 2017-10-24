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

        self.work_dir = None
        self.dataset_cache_dir = None
        self.embedding_dir = None
        self.summaries_dir = None

        self.saver_wrapper = None

        self.dataset_neg = YelpSentences(positive=False,
                                         limit_sentences=self.config['sentence']['limit'],
                                         validation_limit_sentences=self.config['sentence']['validation_limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(),
                                         dataset_name='neg')
        self.dataset_pos = YelpSentences(positive=True,
                                         limit_sentences=self.config['sentence']['limit'],
                                         validation_limit_sentences=self.config['sentence']['validation_limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(),
                                         dataset_name='pos')
        datasets = [self.dataset_neg, self.dataset_pos]
        self.embedding_handler = PreTrainedEmbeddingHandler(
            self.get_embedding_dir(),
            datasets,
            self.config['embedding']['word_size'],
            self.config['embedding']['min_word_occurrences']
        )

        contents, validation_contents = MultiBatchIterator.preprocess(datasets)
        # iterators
        self.batch_iterator = MultiBatchIterator(contents,
                                                 self.embedding_handler,
                                                 self.config['sentence']['min_length'],
                                                 self.config['trainer']['batch_size'])

        # iterators
        self.batch_iterator_validation = MultiBatchIterator(validation_contents,
                                                            self.embedding_handler,
                                                            self.config['sentence']['min_length'],
                                                            self.config['trainer']['validation_batch_size'])
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

    def transfer_batch(self, sess, batch, epoch_num, return_result_as_summary=True, print_to_file=False):
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
        if print_to_file:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_name = os.path.join('logs', '{}-epoch-{}.log'.format(now, epoch_num))
            original_source_strings, transferred_strings = self.print_to_file(
                original_source,
                transferred,
                self.embedding_handler,
                log_file_name
            )
        else:
            # print the reconstruction
            original_target_strings, reconstructed_strings = self.print_side_by_side(
                original_target,
                reconstructed,
                'original_target: ',
                'reconstructed: ',
                self.embedding_handler
            )
            # print the transfer
            original_source_strings, transferred_strings = self.print_side_by_side(
                original_source,
                transferred,
                'original_source: ',
                'transferred: ',
                self.embedding_handler
            )
        #evaluate the transfer
        evaluation_prediction, evaluation_confidence = classify([' '.join(s) for s in transferred_strings])
        evaluation_accuracy = Counter(evaluation_prediction)['pos'] / float(len(evaluation_prediction))
        average_evaluation_confidence = sum(evaluation_confidence) / float(len(evaluation_confidence))
        if print_to_file:
            with open(os.path.join('logs', 'accuracy.log'), 'a+') as f:
                f.write('Date: {}, Epoch: {}, Acc: {}, Confidence: {}\n'.format(now, epoch_num, evaluation_accuracy,
                                                                                average_evaluation_confidence))
        else:
            print('Transferred evaluation acc: {} with average confidence of: {}'.format(
                evaluation_accuracy, average_evaluation_confidence)
            )

        if return_result_as_summary:
            return sess.run(
                self.model.evaluation_summary,
                {
                    self.model.text_watcher.placeholders['original_source']: [' '.join(s) for s in
                                                                              original_source_strings],
                    self.model.text_watcher.placeholders['original_target']: [' '.join(s) for s in
                                                                              original_target_strings],
                    self.model.text_watcher.placeholders['transferred']: [' '.join(s) for s in transferred_strings],
                    self.model.text_watcher.placeholders['reconstructed']: [' '.join(s) for s in reconstructed_strings],
                })
        else:
            return None

    def print_side_by_side(self, batch1, batch2, message1, message2, embedding_handler):
        translated_batch1 = embedding_handler.get_index_to_word(batch1)
        translated_batch2 = embedding_handler.get_index_to_word(batch2)
        # for i in range(len(translated_batch1)):
        #     print(message1)
        #     print(translated_batch1[i])
        #     print(message2)
        #     print(translated_batch2[i])
        return translated_batch1, translated_batch2

    def print_to_file(self, batch1, batch2, embedding_handler, file_name):
        translated_batch1 = embedding_handler.get_index_to_word(batch1)
        translated_batch2 = embedding_handler.get_index_to_word(batch2)
        with open(file_name, 'w') as f:
            for i in range(len(translated_batch1)):
                f.write('O: {}\n'.format(' '.join(translated_batch1[i])))
                f.write('T: {}\n\n'.format(' '.join(translated_batch2[i])))
        return translated_batch1, translated_batch2

    @staticmethod
    def remove_by_length(sentences, lengths):
        return [[
            word for word_index, word in enumerate(sentence) if word_index < lengths[sentence_index]
        ] for sentence_index, sentence in enumerate(sentences)]

    def do_train_loop(self):
        self.saver_wrapper = SaverWrapper(self.get_work_dir(), self.get_trainer_name())
        session_config = tf.ConfigProto(log_device_placement=self.operational_config['print_device'],
                                        allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        if self.operational_config['run_optimizer']:
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=session_config) as sess:
            use_tensorboard = self.operational_config['tensorboard_frequency'] > 0
            if use_tensorboard:
                summary_writer_train = tf.summary.FileWriter(os.path.join(self.get_summaries_dir(), 'train'),
                                                             sess.graph)
                summary_writer_validation = tf.summary.FileWriter(os.path.join(self.get_summaries_dir(), 'validation'))

            sess.run(tf.global_variables_initializer())
            if self.operational_config['load_model']:
                self.saver_wrapper.load_model(sess)

            self.do_before_train_loop(sess)

            global_step = 0
            for epoch_num in range(self.config['trainer']['number_of_epochs']):
                print('epoch {} of {}'.format(epoch_num + 1, self.config['trainer']['number_of_epochs']))
                self.do_before_epoch(sess, global_step, epoch_num)
                for batch_index, batch in enumerate(self.batch_iterator):
                    extract_summaries = use_tensorboard and \
                                        (global_step % self.operational_config['tensorboard_frequency'] == 0)
                    train_summaries = self.do_train_batch(sess, global_step, epoch_num, batch_index, batch,
                                                          extract_summaries=extract_summaries)
                    if train_summaries and extract_summaries:
                        summary_writer_train.add_summary(train_summaries, global_step=global_step)
                    if (global_step % self.operational_config['validation_batch_frequency']) == 1:
                        for validation_batch in self.batch_iterator_validation:
                            if use_tensorboard:
                                validation_summaries = self.do_validation_batch(sess, global_step, epoch_num, batch_index,
                                                                                validation_batch, print_to_file=False)
                                if validation_summaries:
                                    summary_writer_validation.add_summary(validation_summaries, global_step=global_step)
                            break
                    global_step += 1
                self.do_after_epoch(sess, global_step, epoch_num)
            self.do_after_train_loop(sess)

    def do_before_train_loop(self, sess):
        sess.run(self.model.embedding_container.assign_embedding(), {
            self.model.embedding_container.embedding_placeholder: self.embedding_handler.embedding_np
        })

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch, extract_summaries=False):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.target_batch: batch[1].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.target_lengths: batch[1].lengths,
            self.model.dropout_placeholder: self.config['model']['dropout'],
            self.model.discriminator_dropout_placeholder: self.config['model']['discriminator_dropout'],
        }
        print('batch len: {}'.format(batch[0].get_len()))
        execution_list = [
            self.model.master_step,
            self.model.discriminator_loss,
            self.model.accuracy,
            self.model.train_generator,
            self.model.summary_step
        ]
        if extract_summaries:
            _, discriminator_loss, accuracy, train_generator_flag, summary = sess.run(execution_list, feed_dict)
        else:
            _, discriminator_loss, accuracy, train_generator_flag = sess.run(execution_list[:-1], feed_dict)
            summary = None
        print('accuracy: {}'.format(accuracy))
        print('discriminator_loss: {}'.format(discriminator_loss))
        print('training generator? {}'.format(train_generator_flag))
        return summary

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, batch, print_to_file):
        return self.transfer_batch(sess, batch, epoch_num, return_result_as_summary=not print_to_file,
                                   print_to_file=print_to_file)

    def do_after_train_loop(self, sess):
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        print('model loaded, sample sentences:')
        for batch in self.batch_iterator_validation:
            self.transfer_batch(sess, batch, 0, return_result_as_summary=False, print_to_file=False)
            break

    def do_before_epoch(self, sess, global_step, epoch_num):
        sess.run(self.model.epoch_counter.update)

    def do_after_epoch(self, sess, global_step, epoch_num):
        if epoch_num % 10 == 0:
            # activate the saver
            self.saver_wrapper.save_model(sess, global_step=global_step)

if __name__ == "__main__":
    with open("config/gan.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)
    init_logger()
    print('------------ Config ------------')
    print(yaml.dump(config))
    print('------------ Operational Config ------------')
    print(yaml.dump(operational_config))
    ModelTrainer(config, operational_config).do_train_loop()
