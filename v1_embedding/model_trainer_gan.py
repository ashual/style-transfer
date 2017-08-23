import yaml
from datasets.multi_batch_iterator import MultiBatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.contant_iteration_policy import ConstantIterationPolicy
from v1_embedding.convergence_policy import ConvergencePolicy
from v1_embedding.gan_model import GanModel
from v1_embedding.model_trainer_base import ModelTrainerBase
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler


class ModelTrainerGan(ModelTrainerBase):
    def __init__(self, config_file, operational_config_file):
        ModelTrainerBase.__init__(self, config_file=config_file, operational_config_file=operational_config_file)

        self.dataset_neg = YelpSentences(positive=False, limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(), dataset_name='neg')
        self.dataset_pos = YelpSentences(positive=True, limit_sentences=self.config['sentence']['limit'],
                                         dataset_cache_dir=self.get_dataset_cache_dir(), dataset_name='pos')
        datasets = [self.dataset_neg, self.dataset_pos]
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.get_embedding_dir(),
            datasets,
            self.config['embedding']['word_size'],
            self.config['embedding']['min_word_occurrences']
        )

        # iterators
        self.batch_iterator = MultiBatchIterator(datasets,
                                                 self.embedding_handler,
                                                 self.config['sentence']['min_length'],
                                                 self.config['trainer']['batch_size'])

        # iterators
        self.batch_iterator_validation = MultiBatchIterator(datasets,
                                                            self.embedding_handler,
                                                            self.config['sentence']['min_length'],
                                                            2)
        # train loop parameters:
        # self.policy = ConvergencePolicy()
        self.policy = ConstantIterationPolicy(generator_steps=self.config['trainer']['min_generator_steps'],
                                              discriminator_steps=self.config['trainer']['min_discriminator_steps'])

        # set the model
        self.model = GanModel(self.config, self.operational_config, self.embedding_handler)

    def get_trainer_name(self):
        return '{}_{}_{}'.format(self.__class__.__name__, self.config['model']['discriminator_type'],
                                 self.config['model']['loss_type'])

    def do_generator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary, extract_summaries):
        print('started generator')
        # print('running loss: {}'.format(self.policy.running_loss))  # TODO: remove
        execution_list = [
            # self.model.prediction,
            self.model.discriminator_loss,
            self.model.accuracy
        ]
        loss, acc = sess.run(execution_list, feed_dictionary)
        # pred, loss, acc = sess.run(execution_list, feed_dictionary)
        # print('pred: {}'.format(pred))
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        # if self.policy.should_train_generator(global_step, epoch_num, batch_index, pred, loss, acc):
        if self.policy.should_train_generator(global_step, epoch_num, batch_index, None, loss, acc):
            # the generator is still improving
            # print('new running loss: {}'.format(self.policy.running_loss))  # TODO: remove
            print()
            execution_list = [self.model.generator_train_step, self.model.generator_step_summaries]
            if extract_summaries:
                _, s = sess.run(execution_list, feed_dictionary)
                return s
            else:
                sess.run(execution_list[:-1], feed_dictionary)
                return None
        else:
            print('generator too good - training discriminator')
            print()
            # activate the saver
            # self.saver_wrapper.save_model(sess, global_step=global_step)
            # the generator is no longer improving, will train discriminator next
            self.policy.do_train_switch(start_training_generator=False)
            return self.do_discriminator_train(sess, global_step, epoch_num, batch_index, feed_dictionary,
                                               extract_summaries=extract_summaries)

    def do_discriminator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary, extract_summaries):
        print('started discriminator')
        # print('running loss: {}'.format(self.policy.running_loss))  # TODO: remove
        execution_list = [
            # self.model.prediction,
            self.model.discriminator_loss,
            self.model.accuracy
        ]
        # pred, loss, acc = sess.run(execution_list, feed_dictionary)
        loss, acc = sess.run(execution_list, feed_dictionary)
        # print('pred: {}'.format(pred))
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        # if self.policy.should_train_discriminator(global_step, epoch_num, batch_index, pred, loss, acc):
        if self.policy.should_train_discriminator(global_step, epoch_num, batch_index, None, loss, acc):
            # the discriminator is still improving
            # print('new running loss: {}'.format(self.policy.running_loss))  # TODO: remove
            print()
            execution_list = [self.model.discriminator_train_step, self.model.discriminator_step_summaries]
            if extract_summaries:
                _, s = sess.run(execution_list, feed_dictionary)
                return s
            else:
                sess.run(execution_list[:-1], feed_dictionary)
                return None
        else:
            print('discriminator too good - training generator')
            print()
            # the discriminator is no longer improving, will train generator next
            self.policy.do_train_switch(start_training_generator=True)
            return self.do_generator_train(sess, global_step, epoch_num, batch_index, feed_dictionary,
                                           extract_summaries=extract_summaries)

    def transfer_batch(self, sess, batch, return_result_as_summary=True):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.dropout_placeholder: 0.0,
            self.model.discriminator_dropout_placeholder: 0.0,
        }
        transferred_result = sess.run(self.model.transferred_source_batch, feed_dict)
        end_of_sentence_index = self.embedding_handler.word_to_index[self.embedding_handler.end_of_sentence_token]
        # original without paddings:
        original = self.remove_by_length(batch[0].sentences, batch[0].lengths)
        # only take the prefix before EOS:
        transferred = []
        for s in transferred_result:
            if end_of_sentence_index in s:
                transferred.append(s[:s.tolist().index(end_of_sentence_index) + 1])
            else:
                transferred.append(s)
        # print the transfer
        original_strings, transferred_strings = self.print_side_by_side(
            original,
            transferred,
            'original: ',
            'transferred: ',
            self.embedding_handler
        )
        if return_result_as_summary:
            # output validation summary for first 5 sentences
            to_print = 5
            return sess.run(self.model.text_watcher.summary, {
                self.model.text_watcher.placeholder1: [' '.join(s) for s in original_strings[:to_print]],
                self.model.text_watcher.placeholder2: [' '.join(s) for s in transferred_strings[:to_print]],
            })
        else:
            return None

    def do_before_train_loop(self, sess):
        sess.run(self.model.embedding_container.assign_embedding(), {
            self.model.embedding_container.embedding_placeholder: self.embedding_handler.embedding_np
        })
        self.policy.do_train_switch(start_training_generator=False)

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
        if self.policy.train_generator:
            # should train the generator
            return self.do_generator_train(sess, global_step, epoch_num, batch_index, feed_dict,
                                           extract_summaries=extract_summaries)
        else:
            # should train discriminator
            return self.do_discriminator_train(sess, global_step, epoch_num, batch_index, feed_dict,
                                               extract_summaries=extract_summaries)

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, batch):
        return self.transfer_batch(sess, batch, return_result_as_summary=True)

    def do_after_train_loop(self, sess):
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        print('model loaded, sample sentences:')
        for batch in self.batch_iterator_validation:
            self.transfer_batch(sess, batch, return_result_as_summary=False)
            break

    def do_before_epoch(self, sess, global_step, epoch_num):
        sess.run(self.model.assign_epoch, {self.model.epoch_placeholder: epoch_num})

    def do_after_epoch(self, sess, global_step, epoch_num):
        pass


if __name__ == "__main__":
    with open("config/gan.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)

    ModelTrainerGan(config, operational_config).do_train_loop()
