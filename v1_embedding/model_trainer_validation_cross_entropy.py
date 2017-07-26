import os
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
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator


class ModelTrainerValidation(BaseModel):
    def __init__(self, config_file):
        BaseModel.__init__(self)
        self.saver_dir = os.path.join(os.getcwd(), 'models', 'validation_cross_entropy')
        self.saver_path = os.path.join(self.saver_dir, 'validation_cross_entropy')
        self.embedding_dir = os.path.join(self.saver_dir, 'embedding')
        self.summaries_dir = os.path.join(self.saver_dir, 'tensorboard')

        self.config = config_file
        translation_hidden_size = config['translation_hidden_size']

        self.dataset = YelpSentences(positive=False, limit_sentences=config['limit_sentences'])
        self.embedding_handler = WordIndexingEmbeddingHandler(self.embedding_dir, self.dataset,
                                                              config['word_embedding_size'])

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                        translation_hidden_size,
                                                        config['train_embeddings'],
                                                        )
        self.dropout = tf.placeholder(tf.float32, shape=())
        self.encoder = EmbeddingEncoder(config['encoder_hidden_states'], translation_hidden_size, config['dropout'], config['bidirectional'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(), config['decoder_hidden_states'],
                                        self.embedding_translator, self.dropout)
        self.discriminator = EmbeddingDiscriminator(config['discriminator_hidden_states'], translation_hidden_size, config['discriminator_dropout'])
        self.loss_handler = LossHandler()

        self.batch_iterator = BatchIterator(self.dataset, self.embedding_handler,
                                            sentence_len=config['sentence_length'], batch_size=config['batch_size'])

        self.batch_iterator_validation = BatchIterator(self.dataset, self.embedding_handler,
                                                       sentence_len=config['sentence_length'], batch_size=1000)

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

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
        self.train_summaries = tf.summary.merge([loss_summary, accuracy_summary, weight_summaries, gradient_summaries,
                                                 gradient_global_norm])
        self.validation_summaries = tf.summary.merge([accuracy_summary, weight_summaries])

    def overfit(self):
        def print_side_by_side(original, reconstructed):
            translated_original = self.embedding_handler.get_index_to_word(original)
            translated_reconstructed = self.embedding_handler.get_index_to_word(reconstructed)
            for i in range(len(translated_original)):
                print('original:')
                print(translated_original[i])
                print('reconstructed:')
                print(translated_reconstructed[i])

        saver = tf.train.Saver()
        best_validation_acc = -1.0

        if not os.path.exists(self.saver_dir):
                os.makedirs(self.saver_dir)
        print('models are saved to: {}'.format(self.saver_dir))
        print()

        with tf.Session() as sess:
            train_summaries_path = os.path.join(self.summaries_dir, 'train')
            validation_summaries_path = os.path.join(self.summaries_dir, 'validation')
            summary_writer_train = tf.summary.FileWriter(train_summaries_path, sess.graph)
            summary_writer_validation = tf.summary.FileWriter(validation_summaries_path)
            sess.run(tf.global_variables_initializer())
            checkpoint_path = tf.train.get_checkpoint_state(self.saver_dir)
            if config['load_model'] and checkpoint_path is not None:
                saver.restore(sess, checkpoint_path.model_checkpoint_path)
                print('Model restored from file: {}'.format(checkpoint_path.model_checkpoint_path))

            sess.run(self.embedding_translator.assign_embedding(), {
                self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
            })

            global_step = 0
            for epoch_num in range(config['number_of_epochs']):
                print('epoch {} of {}'.format(epoch_num+1, config['number_of_epochs']))

                for i, batch in enumerate(self.batch_iterator):
                    feed_dict = {
                        self.source_batch: batch,
                        self.target_batch: batch,
                        self.dropout: config['dropout'],
                        self.encoder.should_print: self.config['debug'],
                        self.decoder.should_print: self.config['debug'],
                        self.loss_handler.should_print: self.config['debug']
                    }
                    _, loss_output, decoded_output, batch_acc, s = sess.run([self.train_step, self.loss, self.outputs,
                                                                             self.accuracy, self.train_summaries],
                                                                            feed_dict)
                    summary_writer_train.add_summary(s, global_step=global_step)

                    # Validation
                    if i % 100 == 0:
                        print_side_by_side(batch, decoded_output)
                        print('epoch-index: {} batch-index: {} acc: {} loss: {}'.format(epoch_num, i, batch_acc,
                                                                                        loss_output))
                        print()
                        for validation_batch in self.batch_iterator_validation:
                            feed_dict = {
                                self.source_batch: validation_batch,
                                self.target_batch: validation_batch,
                                self.dropout: 1,
                                self.encoder.should_print: self.config['debug'],
                                self.decoder.should_print: self.config['debug'],
                                self.loss_handler.should_print: self.config['debug']
                            }
                            validation_acc, s = sess.run([self.accuracy, self.validation_summaries], feed_dict)
                            break
                        summary_writer_validation.add_summary(s, global_step=global_step)

                        if validation_acc > best_validation_acc:
                            print('saving model, former best accuracy {} current best accuracy {}'.
                                  format(best_validation_acc, validation_acc))
                            try:
                                # save model
                                saver.save(sess, self.saver_path)
                                best_validation_acc = validation_acc
                                print('Model saved')
                                print()
                            except:
                                print('Failed to save model')
                                print()
                    global_step += 1

            print('best validation accuracy: {}'.format(best_validation_acc))
            # make sure the model is correct:
            saver.restore(sess, self.saver_path)
            for validation_batch in self.batch_iterator_validation:
                feed_dict = {
                    self.source_batch: validation_batch,
                    self.target_batch: validation_batch,
                    self.encoder.should_print: self.config['debug'],
                    self.decoder.should_print: self.config['debug'],
                    self.loss_handler.should_print: self.config['debug']
                }
                validation_acc = sess.run(self.accuracy, feed_dict)
                break
            print('tested validation accuracy: {}'.format(validation_acc))

if __name__ == "__main__":
    with open("config/validation_word_index.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)

    ModelTrainerValidation(config).overfit()
