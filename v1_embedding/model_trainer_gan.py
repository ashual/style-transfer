import yaml
from datasets.multi_batch_iterator import MultiBatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.convergence_policy import ConvergencePolicy
from v1_embedding.gan_model_content import GanModelContent
from v1_embedding.gan_model_embedding import GanModelEmbedding
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
                                                 self.config['model']['batch_size'])

        # iterators
        self.batch_iterator_validation = MultiBatchIterator(datasets,
                                                            self.embedding_handler,
                                                            self.config['sentence']['min_length'],
                                                            2)
        # train loop parameters:
        self.policy = ConvergencePolicy()

        # set the model
        if self.config['model']['discriminator_type'] == 'embedding':
            self.model = GanModelEmbedding(self.config, self.operational_config, self.embedding_handler)
        elif self.config['model']['discriminator_type'] == 'content':
            self.model = GanModelContent(self.config, self.operational_config, self.embedding_handler)

        self.discriminator_step_summaries, self.generator_step_summaries = self.model.create_summaries()

    def get_trainer_name(self):
        return '{}_{}'.format(self.__class__.__name__, self.config['model']['discriminator_type'])

    def do_generator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary, extract_summaries):
        print('started generator')
        print('running loss: {}'.format(self.policy.running_loss))  # TODO: remove
        execution_list = [
            self.model.generator_step_prediction,
            self.model.discriminator_loss_on_generator_step,
            self.model.discriminator_accuracy_for_generator
        ]
        pred, loss, acc = sess.run(execution_list, feed_dictionary)
        print('pred: {}'.format(pred))
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        if self.policy.should_train_generator(global_step, epoch_num, batch_index, pred, loss, acc):
            # the generator is still improving
            print('new running loss: {}'.format(self.policy.running_loss))  # TODO: remove
            print()
            execution_list = [self.model.generator_train_step, self.generator_step_summaries]
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
            self.saver_wrapper.save_model(sess, global_step=global_step)
            # the generator is no longer improving, will train discriminator next
            self.policy.do_train_switch(start_training_generator=False)
            sess.run(self.model.assign_train_generator, {self.model.train_generator_placeholder: 0})
            return self.do_discriminator_train(sess, global_step, epoch_num, batch_index, feed_dictionary,
                                               extract_summaries=extract_summaries)

    def do_discriminator_train(self, sess, global_step, epoch_num, batch_index, feed_dictionary, extract_summaries):
        print('started discriminator')
        print('running loss: {}'.format(self.policy.running_loss))  # TODO: remove
        execution_list = [
            self.model.discriminator_step_prediction,
            self.model.discriminator_loss_on_discriminator_step,
            self.model.discriminator_accuracy_for_discriminator
        ]
        pred, loss, acc = sess.run(execution_list, feed_dictionary)
        print('pred: {}'.format(pred))
        print('acc: {}'.format(acc))
        print('loss: {}'.format(loss))
        if self.policy.should_train_discriminator(global_step, epoch_num, batch_index, pred, loss, acc):
            # the discriminator is still improving
            print('new running loss: {}'.format(self.policy.running_loss))  # TODO: remove
            print()
            execution_list = [self.model.discriminator_train_step, self.discriminator_step_summaries]
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
            sess.run(self.model.assign_train_generator, {self.model.train_generator_placeholder: 1})
            return self.do_generator_train(sess, global_step, epoch_num, batch_index, feed_dictionary,
                                           extract_summaries=extract_summaries)

    def transfer_batch(self, sess, batch):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.dropout_placeholder: 0.0,
            self.model.discriminator_dropout_placeholder: 0.0,
            self.model.encoder.should_print: self.operational_config['debug'],
            self.model.decoder.should_print: self.operational_config['debug'],
            self.model.discriminator.should_print: self.operational_config['debug'],
            self.model.embedding_translator.should_print: self.operational_config['debug'],
        }
        transferred_result = sess.run(self.model.transfer, feed_dict)
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
        self.print_side_by_side(
            original,
            transferred,
            'original: ',
            'transferred: ',
            self.embedding_handler
        )

    def do_before_train_loop(self, sess):
        sess.run(self.model.embedding_translator.assign_embedding(), {
            self.model.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
        })
        self.policy.do_train_switch(start_training_generator=False)
        sess.run(self.model.assign_train_generator, {self.model.train_generator_placeholder: 0})

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch, extract_summaries=False):
        feed_dict = {
            self.model.source_batch: batch[0].sentences,
            self.model.target_batch: batch[1].sentences,
            self.model.source_lengths: batch[0].lengths,
            self.model.target_lengths: batch[1].lengths,
            self.model.dropout_placeholder: self.config['model']['dropout'],
            self.model.discriminator_dropout_placeholder: self.config['model']['discriminator_dropout'],
            self.model.encoder.should_print: self.operational_config['debug'],
            self.model.decoder.should_print: self.operational_config['debug'],
            self.model.discriminator.should_print: self.operational_config['debug'],
            self.model.embedding_translator.should_print: self.operational_config['debug'],
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
        self.transfer_batch(sess, batch)

    def do_after_train_loop(self, sess):
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        print('model loaded, sample sentences:')
        for batch in self.batch_iterator_validation:
            self.transfer_batch(sess, batch)
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
