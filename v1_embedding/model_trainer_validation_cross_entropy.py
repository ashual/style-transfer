import yaml
import tensorflow as tf
import time
from datasets.batch_iterator import BatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.model_trainer_base import ModelTrainerBase


class ModelTrainerValidation(ModelTrainerBase):
    def __init__(self, config_file, operational_config_file):
        ModelTrainerBase.__init__(self, config_file=config_file, operational_config_file=operational_config_file)

        self.best_validation_acc = tf.Variable(-1.0, trainable=False)

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.dataset = YelpSentences(positive=False, limit_sentences=self.config['sentence']['limit'],
                                     dataset_cache_dir=self.dataset_cache_dir)
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.embedding_dir,
            [self.dataset],
            self.config['embedding']['word_size'],
            self.config['embedding']['min_word_occurrences']
        )
        self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                        self.config['model']['translation_hidden_size'],
                                                        self.config['embedding']['should_train'])
        self.encoder = EmbeddingEncoder(self.config['model']['encoder_hidden_states'],
                                        self.dropout_placeholder,
                                        self.config['model']['bidirectional_encoder'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['model']['decoder_hidden_states'],
                                        self.embedding_translator, self.dropout_placeholder)

        # "One Hot Vector" -> Embedded Vector (w2v)
        embeddings = self.embedding_translator.embed_inputs(self.source_batch)
        # Embedded Vector (w2v) -> Encoded (constant length)
        encoded = self.encoder.encode_inputs_to_vector(embeddings, self.source_identifier)
        # Encoded -> Decoded
        partial_embedding = embeddings[:, :-1, :]
        decoded = self.decoder.do_teacher_forcing(encoded, partial_embedding, self.source_identifier)
        # decoded -> logits
        logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(decoded)
        # cross entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.source_batch, tf.shape(logits)[-1]), logits=logits))
        # training
        optimizer = tf.train.GradientDescentOptimizer(self.config['model']['learn_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss, colocate_gradients_with_ops=True)
        self.train_step = optimizer.apply_gradients(grads_and_vars)
        # maximal word for each step
        self.outputs = self.embedding_translator.translate_logits_to_words(logits)
        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.source_batch, self.outputs), tf.float32))
        self.best_loss = float('inf')
        self.loss_output = float('inf')
        # summaries
        loss_summary = tf.summary.scalar('loss', self.loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
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
                                                 gradient_global_norm])
        self.validation_summaries = tf.summary.merge([accuracy_summary, weight_summaries, loss_summary])

    def print_side_by_side(self, original, reconstructed):
        translated_original = self.embedding_handler.get_index_to_word(original)
        translated_reconstructed = self.embedding_handler.get_index_to_word(reconstructed)
        for i in range(len(translated_original)):
            print('original:')
            print(translated_original[i])
            print('reconstructed:')
            print(translated_reconstructed[i])

    def do_before_train_loop(self, sess):
        best_validation_acc = sess.run(self.best_validation_acc)
        print('starting validation accuracy: {}'.format(best_validation_acc))
        sess.run(self.embedding_translator.assign_embedding(), {
            self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
        })

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch):
        feed_dict = {
            self.source_batch: batch,
            self.target_batch: batch,
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
            self.print_side_by_side(batch, decoded_output)
            print('epoch-index: {} batch-index: {} acc: {} loss: {} runtime: {}'.format(epoch_num, batch_index,
                                                                                        batch_acc, loss_output,
                                                                                        total_time))
            print()
        else:
            # will not run summaries
            _, loss_output, decoded_output, batch_acc = sess.run(execution_list[:-1], feed_dict)

        return train_summaries

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, validation_batch):
        feed_dict = {
            self.source_batch: validation_batch,
            self.target_batch: validation_batch,
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
        for validation_batch in self.batch_iterator_validation:
            feed_dict = {
                self.source_batch: validation_batch,
                self.target_batch: validation_batch,
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
