import os
import tensorflow as tf
from v1_embedding.saver_wrapper import SaverWrapper


class ModelTrainerBase:
    def __init__(self, config_file, operational_config_file):
        self.config = config_file
        self.operational_config = operational_config_file

        self.work_dir = os.path.join(os.getcwd(), 'models', self.get_trainer_name())
        self.dataset_cache_dir = os.path.join(self.work_dir, 'dataset_cache')
        self.embedding_dir = os.path.join(self.work_dir, 'embedding')
        self.summaries_dir = os.path.join(self.work_dir, 'tensorboard')

        self.batch_iterator = None
        self.batch_iterator_validation = None

        self.saver_wrapper = None

    def get_trainer_name(self):
        return self.__class__.__name__

    def do_train_loop(self):
        self.saver_wrapper = SaverWrapper(self.work_dir, self.get_trainer_name())
        session_config = tf.ConfigProto(log_device_placement=self.operational_config['print_device'],
                                        allow_soft_placement=True)
        if self.operational_config['run_optimizer']:
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=session_config) as sess:
            summary_writer_train = tf.summary.FileWriter(os.path.join(self.summaries_dir, 'train'), sess.graph)
            summary_writer_validation = tf.summary.FileWriter(os.path.join(self.summaries_dir, 'validation'))

            sess.run(tf.global_variables_initializer())
            if self.operational_config['load_model']:
                self.saver_wrapper.load_model(sess)

            self.do_before_train_loop(sess)

            global_step = 0
            for epoch_num in range(self.config['number_of_epochs']):
                print('epoch {} of {}'.format(epoch_num+1, self.config['number_of_epochs']))

                for batch_index, batch in enumerate(self.batch_iterator):
                    train_summaries = self.do_train_batch(sess, global_step, epoch_num, batch_index, batch)
                    if train_summaries is not None:
                        summary_writer_train.add_summary(train_summaries, global_step=global_step)
                    if batch_index % 100 == 0:
                        for validation_batch in self.batch_iterator_validation:
                            validation_summaries = self.do_validation_batch(sess, global_step, epoch_num, batch_index,
                                                                            validation_batch)
                            summary_writer_validation.add_summary(validation_summaries, global_step=global_step)
                            break
                    global_step += 1
            self.do_after_train_loop(sess)

    def do_before_train_loop(self, sess):
        pass

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch):
        pass

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, validation_batch):
        pass

    def do_after_train_loop(self, sess):
        pass
