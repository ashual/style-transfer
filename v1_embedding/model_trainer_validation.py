import numpy as np
import yaml
import tensorflow as tf
import os
from datasets.batch_iterator import BatchIterator
from datasets.yelp_helpers import YelpSentences
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.glove_embedding_handler import GloveEmbeddingHandler
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.loss_handler import LossHandler
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator


class ModelTrainerValidation(BaseModel):
    def __init__(self, config_file):
        BaseModel.__init__(self)
        self.config = config_file
        translation_hidden_size = config['translation_hidden_size']
        self.saver_dir = os.path.join(os.getcwd(), 'models', 'validation')
        self.saver_path = os.path.join(self.saver_dir, 'validation')
        self.embedding_dir = os.path.join(self.saver_dir, 'embedding')

        self.dataset = YelpSentences(positive=False, limit_sentences=config['limit_sentences'])
        self.vocabulary_handler = GloveEmbeddingHandler(
            save_dir=self.embedding_dir,
            pretrained_glove_file=config['glove_file'],
            force_vocab=False,
            dataset=self.dataset
        )

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int32, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int32, shape=(None, None))

        self.embedding_translator = EmbeddingTranslator(
            self.vocabulary_handler,
            translation_hidden_size,
            config['train_embeddings'],
        )
        self.encoder = EmbeddingEncoder(
            config['encoder_hidden_states'],
            translation_hidden_size
        )
        self.decoder = EmbeddingDecoder(
            self.vocabulary_handler.get_embedding_size(),
            config['decoder_hidden_states'],
            self.embedding_translator
        )
        self.discriminator = EmbeddingDiscriminator(
            config['discriminator_hidden_states'],
            translation_hidden_size
        )
        self.loss_handler = LossHandler()

        self.batch_iterator = BatchIterator(
            self.dataset,
            self.vocabulary_handler,
            sentence_len=config['sentence_length'],
            batch_size=config['batch_size']
        )

    def overfit(self):
        def decoded_to_vocab(decoded):
            square = np.square(self.vocabulary_handler.embedding_np - decoded)
            dist = np.sum(square, axis=1)
            best_index = np.argmin(dist, 0)
            return best_index

        def decoded_sentences_to_vocab(sentences_indices):
            indices = [[decoded_to_vocab(x) for x in r] for r in sentences_indices]
            return self.vocabulary_handler.get_index_to_word(indices)

        train_step, loss, decoded = self.create_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []

            sess.run(self.embedding_translator.assign_embedding(), {
                self.embedding_translator.embedding_placeholder: self.vocabulary_handler.embedding_np
            })

            for epoch_num in range(1000):
                for i, batch in enumerate(self.batch_iterator):
                    feed_dict = {
                        self.source_batch: batch,
                        self.target_batch: batch,
                        self.encoder.should_print: self.config['debug'],
                        self.decoder.should_print: self.config['debug'],
                        self.loss_handler.should_print: self.config['debug']
                    }
                    _, loss_output, decoded_output = sess.run([train_step, loss, decoded], feed_dict)

                    string_output = decoded_sentences_to_vocab(decoded_output)
                    print('batch-index {} loss {}'.format(i, loss_output))
                    print('original: {}'.format([[self.vocabulary_handler.index_to_word[x] for x in s] for s in batch]))
                    print('reconstructed: {}'.format(string_output))
                    print()
                    training_losses.append(loss_output)

    def create_model(self, ):
        # "One Hot Vector" -> Embedded Vector (w2v)
        embeddings = self.embedding_translator.embed_inputs(self.source_batch)
        # Embedded Vector (w2v) -> Encoded (constant length)
        encoded = self.encoder.encode_inputs_to_vector(embeddings, self.source_identifier)
        # Encoded -> Decoded
        partial_embedding = embeddings[:, :-1, :]
        decoded = self.decoder.do_teacher_forcing(encoded, partial_embedding, self.source_identifier)
        # Reconstruction Loss (Embedded, Decoded)
        loss = self.loss_handler.get_context_vector_distance_loss(embeddings, decoded)
        train_step = tf.train.AdamOptimizer(self.config['learn_rate']).minimize(loss)
        return train_step, loss, decoded

if __name__ == "__main__":
    with open("config/validation.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)

    ModelTrainerValidation(config).overfit()
