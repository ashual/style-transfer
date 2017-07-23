import os
import yaml
import time
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
        self.config = config_file
        translation_hidden_size = config['translation_hidden_size']

        self.dataset = YelpSentences(positive=False, limit_sentences=config['limit_sentences'])
        self.embedding_handler = WordIndexingEmbeddingHandler(self.dataset, config['word_embedding_size'])

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                        translation_hidden_size,
                                                        config['train_embeddings'],
                                                        )
        self.encoder = EmbeddingEncoder(config['encoder_hidden_states'], translation_hidden_size)
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(), config['decoder_hidden_states'],
                                        self.embedding_translator)
        self.discriminator = EmbeddingDiscriminator(config['discriminator_hidden_states'], translation_hidden_size)
        self.loss_handler = LossHandler()

        self.batch_iterator = BatchIterator(self.dataset, self.embedding_handler,
                                            sentence_len=config['sentence_length'], batch_size=config['batch_size'])

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

    def overfit(self):
        saver = tf.train.Saver()
        last_save_time = time.time()
        saver_dir = os.path.join(os.getcwd(), 'models', 'validation_cross_entropy')
        saver_path = os.path.join(saver_dir, 'validation_cross_entropy')
        if not os.path.exists(saver_dir):
                os.makedirs(saver_dir)
        print('models are saved to: {}'.format(saver_dir))
        print()

        train_step, loss, outputs, acc = self.create_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint_path = tf.train.get_checkpoint_state(saver_dir)
            if config['load_model'] and checkpoint_path is not None:
                saver.restore(sess, checkpoint_path.model_checkpoint_path)
                print('Model restored from file: {}'.format(checkpoint_path.model_checkpoint_path))
            training_losses = []

            sess.run(self.embedding_translator.assign_embedding(), {
                self.embedding_translator.embedding_placeholder: self.embedding_handler.embedding_np
            })

            for epoch_num in range(config['number_of_epochs']):
                print('epoch {} of {}'.format(epoch_num+1, config['number_of_epochs']))

                for i, batch in enumerate(self.batch_iterator):
                    feed_dict = {
                        self.source_batch: batch,
                        self.target_batch: batch,
                        self.encoder.should_print: self.config['debug'],
                        self.decoder.should_print: self.config['debug'],
                        self.loss_handler.should_print: self.config['debug']
                    }
                    _, loss_output, decoded_output, batch_acc = sess.run([train_step, loss, outputs, acc], feed_dict)
                    training_losses.append(loss_output)

                    if i % 100 == 0:
                        print('batch-index: {} acc: {} loss: {}'.format(i, batch_acc, loss_output))
                        print('original:')
                        print(self.embedding_handler.get_index_to_word(batch))
                        print('reconstructed:')
                        print(self.embedding_handler.get_index_to_word(decoded_output))
                        print()

                    if (time.time() - last_save_time) >= (60 * 5):
                        try:
                            # save model
                            saver.save(sess, saver_path)
                            last_save_time = time.time()
                            print('Model saved')
                            print()
                        except:
                            print('Failed to save model')
                            print()

    def create_model(self, ):
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
        loss = self.loss_handler.get_sentence_reconstruction_loss(self.source_batch, logits)
        train_step = tf.train.AdamOptimizer(self.config['learn_rate']).minimize(loss)
        outputs = self.embedding_translator.translate_logits_to_words(logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.source_batch, outputs), tf.float32))
        return train_step, loss, outputs, accuracy

if __name__ == "__main__":
    with open("config/validation_word_index.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)

    ModelTrainerValidation(config).overfit()
