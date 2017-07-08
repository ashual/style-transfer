import tensorflow as tf
from v1_embedding.base_model import BaseModel
from v1_embedding.embedding_translator import EmbeddingTranslator


class ModelTrainer(BaseModel):
    def __init__(self, config, vocabulary_handler, loss_handler, encoder, decoder, discriminator):
        BaseModel.__init__(self)
        self.config = config
        self.vocabulary_handler = vocabulary_handler

        # TODO init the below here (instead of get them as inputs)
        self.loss_handler = loss_handler
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        # end of TODO

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int32, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int32, shape=(None, None))

        self.embedding_translator = EmbeddingTranslator(vocabulary_handler.embedding_size,
                                                        vocabulary_handler.vocabulary_size,
                                                        config.translation_hidden_size,
                                                        vocabulary_handler.start_token_index,
                                                        vocabulary_handler.stop_token_index,
                                                        vocabulary_handler.unknown_token_index,
                                                        vocabulary_handler.pad_token_index)

    def train_discriminator(self):
        target_batch = self.print_tensor_with_shape(self.target_batch, 'target_batch')
        target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        discriminator_prediction_on_target = self.discriminator.encode_inputs_to_vector(target_embbeding)
        target_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_target, True)

        source_batch = self.print_tensor_with_shape(self.source_batch, 'source_batch')
        source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, -1*tf.ones(shape=()))
        sentence_length = tf.shape(target_batch)[1]
        decoded_fake_target = self.decoder.do_iterative_decoding(encoded_source, tf.ones(shape=()),
                                                                 iterations_limit=sentence_length)
        discriminator_prediction_on_fake = self.discriminator.encode_inputs_to_vector(decoded_fake_target)
        fake_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_fake, False)

        total_loss = target_loss + fake_loss
        # TODO add adam optimizer only on discriminator
        pass

    def train_generator(self):
        pass
