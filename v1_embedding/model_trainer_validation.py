import numpy as np
import tensorflow as tf
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.input_utils import InputPipeline, EmbeddingHandler
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.loss_handler import LossHandler
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator

from os import getcwd
from os.path import join


class ModelTrainerValidation(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self)
        self.config = config
        translation_hidden_size = config['translation_hidden_size']

        self.vocabulary_handler = EmbeddingHandler(pretrained_glove_file=config['glove_file'], force_vocab=False,
                                                   start_of_sentence_token='START', end_of_sentence_token='END',
                                                   unknown_token='UNK', pad_token='PAD')

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int32, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int32, shape=(None, None))

        self.embedding_translator = EmbeddingTranslator(self.vocabulary_handler.embedding_size,
                                                        self.vocabulary_handler.vocabulary_size,
                                                        translation_hidden_size,
                                                        config['train_embeddings'],
                                                        self.vocabulary_handler.start_token_index,
                                                        self.vocabulary_handler.stop_token_index,
                                                        self.vocabulary_handler.unknown_token_index,
                                                        self.vocabulary_handler.pad_token_index,
                                                        )
        self.encoder = EmbeddingEncoder(config['encoder_hidden_states'], translation_hidden_size)
        self.decoder = EmbeddingDecoder(self.vocabulary_handler.embedding_size, config['decoder_hidden_states'],
                                        self.embedding_translator)
        self.discriminator = EmbeddingDiscriminator(config['discriminator_hidden_states'], translation_hidden_size)
        self.loss_handler = LossHandler()

        self.input_pipeline = InputPipeline(text_file=config['src_file'], embedding_handler=self.vocabulary_handler,
                                            limit_sentences=config['limit_sentences'])

    def overfit(self):
        def decoded_to_vocab(decoded):
            square = np.abs(self.vocabulary_handler.embedding_np - decoded)
            dist = np.sum(square, axis=1)
            best_index = np.argmin(dist, 0)
            return best_index

        train_step, loss, decoded = self.create_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []

            sess.run(self.embedding_translator.assign_embedding(), {
                self.embedding_translator.embedding_placeholder: self.vocabulary_handler.embedding_np
            })

            for epoch_num in range(1000):
                epoch_iter = self.input_pipeline.batch_iterator(shuffle=True, maximal_batch=self.config['batch_size'])
                for i,(idx, batch) in enumerate(epoch_iter):
                    feed_dict = {
                        self.source_batch: batch,
                        self.target_batch: batch,
                        self.should_print: True
                    }
                    _, loss_output, decoded_output = sess.run([train_step, loss, decoded], feed_dict)

                    string_output = [[self.vocabulary_handler.index_to_word[decoded_to_vocab(x)]
                                      for x in r] for r in decoded_output]
                    print('batch-index {} loss {} reconstructed: {}'.format(i, loss_output, string_output))
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

glove_file = join(getcwd(), "..", "data", "glove.6B", "glove.6B.50d.txt")
src_file = join(getcwd(), "..", "datasets", "yoda", "english.text")

config = {
    'glove_file': glove_file,
    'src_file': src_file,
    'limit_sentences': None,
    'batch_size': 10,
    'translation_hidden_size': 10,
    'train_embeddings': False,
    'encoder_hidden_states': [10, 10],
    'decoder_hidden_states': [10, 10],
    'discriminator_hidden_states': [10, 10],
    'learn_rate': 0.0003,
}
ModelTrainerValidation(config).overfit()
