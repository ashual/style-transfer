import os
import tensorflow as tf
from v1_embedding.saver_wrapper import SaverWrapper


class ModelTrainerBase:
    def __init__(self, config_file, operational_config_file):
        self.config = config_file
        self.operational_config = operational_config_file

        self.work_dir = None
        self.dataset_cache_dir = None
        self.embedding_dir = None
        self.summaries_dir = None

        self.saver_wrapper = None

        # implementations should start the iterators
        self.batch_iterator = None
        self.batch_iterator_validation = None

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
        return self.__class__.__name__

    def print_side_by_side(self, batch1, batch2, message1, message2, embedding_handler):
        translated_batch1 = embedding_handler.get_index_to_word(batch1)
        translated_batch2 = embedding_handler.get_index_to_word(batch2)
        for i in range(len(translated_batch1)):
            print(message1)
            print(translated_batch1[i])
            print(message2)
            print(translated_batch2[i])

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
                summary_writer_train = tf.summary.FileWriter(os.path.join(self.get_summaries_dir(), 'train'), sess.graph)
                summary_writer_validation = tf.summary.FileWriter(os.path.join(self.get_summaries_dir(), 'validation'))

            sess.run(tf.global_variables_initializer())
            if self.operational_config['load_model']:
                self.saver_wrapper.load_model(sess)

            self.do_before_train_loop(sess)

            global_step = 0
            for epoch_num in range(self.config['model']['number_of_epochs']):
                print('epoch {} of {}'.format(epoch_num+1, self.config['model']['number_of_epochs']))
                self.do_before_epoch(sess, global_step, epoch_num)
                for batch_index, batch in enumerate(self.batch_iterator):
                    extract_summaries = use_tensorboard and \
                                        (global_step % self.operational_config['tensorboard_frequency'] == 0)
                    train_summaries = self.do_train_batch(sess, global_step, epoch_num, batch_index, batch,
                                                          extract_summaries=extract_summaries)
                    if train_summaries and extract_summaries:
                        summary_writer_train.add_summary(train_summaries, global_step=global_step)
                    if (global_step % self.operational_config['validation_batch_frequency']) == 0:
                        for validation_batch in self.batch_iterator_validation:
                            validation_summaries = self.do_validation_batch(sess, global_step, epoch_num, batch_index,
                                                                            validation_batch)
                            if validation_summaries and use_tensorboard:
                                summary_writer_validation.add_summary(validation_summaries, global_step=global_step)
                            break
                    global_step += 1
                self.do_after_epoch(sess, global_step, epoch_num)
            self.do_after_train_loop(sess)

    def do_before_train_loop(self, sess):
        pass

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch, extract_summaries=False):
        pass

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, batch):
        pass

    def do_after_train_loop(self, sess):
        pass

    def do_before_epoch(self, sess, global_step, epoch_num):
        pass

    def do_after_epoch(self, sess, global_step, epoch_num):
        pass
