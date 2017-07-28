import yaml
import tensorflow as tf
from datasets.batch_iterator import BatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.loss_handler import LossHandler
from v1_embedding.model_trainer_base import ModelTrainerBase


class ModelTrainerValidation(ModelTrainerBase):
    def __init__(self, config_file):
        ModelTrainerBase.__init__(self, config_file=config_file)

        self.best_validation_acc = None

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.dataset = YelpSentences(positive=False, limit_sentences=self.config['limit_sentences'])
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.embedding_dir,
            self.dataset,
            self.config['word_embedding_size'],
            self.config['min_word_occurrences']
        )
        self.embedding_translator = EmbeddingTranslator(self.embedding_handler, self.config['translation_hidden_size'],
                                                        self.config['train_embeddings'])
        self.encoder = EmbeddingEncoder(self.config['encoder_hidden_states'], self.dropout_placeholder,
                                        self.config['bidirectional_encoder'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['decoder_hidden_states'],
                                        self.embedding_translator, self.dropout_placeholder)
        self.loss_handler = LossHandler()

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
        self.loss = self.loss_handler.get_sentence_reconstruction_loss(self.source_batch, logits)
        # training
        optimizer = tf.train.AdamOptimizer(self.config['learn_rate'])
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_step = optimizer.apply_gradients(grads_and_vars)
        # maximal word for each step
        self.outputs = self.embedding_translator.translate_logits_to_words(logits)
        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.source_batch, self.outputs), tf.float32))
        # summaries
        loss_summary = tf.summary.scalar('loss', self.loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        weight_summaries = tf.summary.merge(self.embedding_translator.get_trainable_parameters_summaries() +
                                            self.encoder.get_trainable_parameters_summaries() +
                                            self.decoder.get_trainable_parameters_summaries())
        gradient_summaries = tf.summary.merge([item for sublist in
                                               [BaseModel.create_summaries(g) for g, v in grads_and_vars]
                                               for item in sublist])
        gradient_global_norm = tf.summary.scalar('gradient_global_norm', tf.global_norm([g for g,v in grads_and_vars]))
        # needed by ModelTrainerBase
        self.batch_iterator = BatchIterator(self.dataset, self.embedding_handler,
                                            sentence_len=self.config['sentence_length'],
                                            batch_size=self.config['batch_size'])

        self.batch_iterator_validation = BatchIterator(self.dataset, self.embedding_handler,
                                                       sentence_len=self.config['sentence_length'], batch_size=1000)

        self.train_summaries = tf.summary.merge([loss_summary, accuracy_summary, weight_summaries, gradient_summaries,
                                                 gradient_global_norm])
        self.validation_summaries = tf.summary.merge([accuracy_summary, weight_summaries])

    def get_trainer_name(self):
        return 'validation_cross_entropy'

    def print_side_by_side(self, original, reconstructed):
        translated_original = self.embedding_handler.get_index_to_word(original)
        translated_reconstructed = self.embedding_handler.get_index_to_word(reconstructed)
        for i in range(len(translated_original)):
            print('original:')
            print(translated_original[i])
            print('reconstructed:')
            print(translated_reconstructed[i])

    def do_before_train_loop(self, sess):
        self.best_validation_acc = -1.0
        sess.run(self.embedding_translator.assign_embedding(), {
            self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
        })

    def do_train_batch(self, sess, global_step, epoch_num, batch_index, batch):
        feed_dict = {
            self.source_batch: batch,
            self.target_batch: batch,
            self.dropout_placeholder: self.config['dropout'],
            self.encoder.should_print: self.config['debug'],
            self.decoder.should_print: self.config['debug'],
            self.loss_handler.should_print: self.config['debug']
        }
        _, loss_output, decoded_output, batch_acc, train_summaries = sess.run([self.train_step, self.loss, self.outputs,
                                                                               self.accuracy, self.train_summaries],
                                                                              feed_dict)
        # print results
        if batch_index % 100 == 0:
            self.print_side_by_side(batch, decoded_output)
            print('epoch-index: {} batch-index: {} acc: {} loss: {}'.format(epoch_num, batch_index, batch_acc,
                                                                            loss_output))
            print()
        return train_summaries

    def do_validation_batch(self, sess, global_step, epoch_num, batch_index, validation_batch,
                            ):
        feed_dict = {
            self.source_batch: validation_batch,
            self.target_batch: validation_batch,
            self.dropout_placeholder: 0.0,
            self.encoder.should_print: self.config['debug'],
            self.decoder.should_print: self.config['debug'],
            self.loss_handler.should_print: self.config['debug']
        }
        validation_acc, validation_summaries = sess.run([self.accuracy, self.validation_summaries], feed_dict)
        if validation_acc > self.best_validation_acc:
            print('saving model, former best accuracy {} current best accuracy {}'.
                  format(self.best_validation_acc, validation_acc))
            print()
            if self.saver_wrapper.save_model(sess):
                self.best_validation_acc = validation_acc

        return validation_summaries

    def do_after_train_loop(self, sess):
        print('best validation accuracy: {}'.format(self.best_validation_acc))
        # make sure the model is correct:
        self.saver_wrapper.load_model(sess)
        for validation_batch in self.batch_iterator_validation:
            feed_dict = {
                self.source_batch: validation_batch,
                self.target_batch: validation_batch,
                self.dropout_placeholder: 0.0,
                self.encoder.should_print: self.config['debug'],
                self.decoder.should_print: self.config['debug'],
                self.loss_handler.should_print: self.config['debug']
            }
            validation_acc = sess.run(self.accuracy, feed_dict)
            print('tested validation accuracy: {}'.format(validation_acc))
            print()
            break

if __name__ == "__main__":
    with open("config/validation_word_index.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)

    ModelTrainerValidation(config).do_train_loop()
