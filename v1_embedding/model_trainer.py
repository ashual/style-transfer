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

GLOVE_FILE = join(getcwd(), "..", "data", "glove.6B", "glove.6B.50d.txt")
SRC_FILE = join(getcwd(), "..", "datasets", "yoda", "english.text")


class ModelTrainer(BaseModel):
    def __init__(self, config, vocabulary_handler,
                 source_reconstruction_loss_coefficient=0, target_reconstruction_loss_coefficient=0,
                 source_professor_loss_coefficient=0, target_professor_loss_coefficient=0,
                 semantic_distance_loss_coefficient=0, anti_discriminator_loss_coefficient=0, ):
        BaseModel.__init__(self)
        self.config = config
        translation_hidden_size = config['translation_hidden_size']

        self.vocabulary_handler = vocabulary_handler

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.source_reconstruction_loss_coefficient = source_reconstruction_loss_coefficient
        self.target_reconstruction_loss_coefficient = target_reconstruction_loss_coefficient
        self.source_professor_loss_coefficient = source_professor_loss_coefficient
        self.target_professor_loss_coefficient = target_professor_loss_coefficient
        self.semantic_distance_loss_coefficient = semantic_distance_loss_coefficient
        self.anti_discriminator_loss_coefficient = anti_discriminator_loss_coefficient

        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int32, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int32, shape=(None, None))

        self.embedding_translator = EmbeddingTranslator(vocabulary_handler.embedding_size,
                                                        vocabulary_handler.vocabulary_size,
                                                        translation_hidden_size,
                                                        vocabulary_handler.start_token_index,
                                                        vocabulary_handler.stop_token_index,
                                                        vocabulary_handler.unknown_token_index,
                                                        vocabulary_handler.pad_token_index,
                                                        self.source_batch)
        self.encoder = EmbeddingEncoder(config['encoder_hidden_states'], translation_hidden_size)
        self.decoder = EmbeddingDecoder(vocabulary_handler.embedding_size, config['decoder_hidden_states'],
                                        self.embedding_translator)
        self.discriminator = EmbeddingDiscriminator(config['discriminator_hidden_states'], translation_hidden_size)
        self.loss_handler = LossHandler()

    def train_discriminator(self):
        target_batch = self.print_tensor_with_shape(self.target_batch, 'target_batch')
        target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        discriminator_prediction_on_target = self.discriminator.encode_inputs_to_vector(target_embbeding)
        target_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_target, True)[0]

        source_batch = self.print_tensor_with_shape(self.source_batch, 'source_batch')
        source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, self.source_identifier)
        sentence_length = tf.shape(target_batch)[1]
        decoded_fake_target = self.decoder.do_iterative_decoding(encoded_source, self.target_identifier,
                                                                 iterations_limit=sentence_length)
        discriminator_prediction_on_fake = self.discriminator.encode_inputs_to_vector(decoded_fake_target)
        fake_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_fake, False)[0]

        total_loss = target_loss + fake_loss
        var_list = self.discriminator.get_trainable_parameters()
        train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(total_loss,
                                                                                              var_list=var_list)
        return train_step, total_loss

    def train_generator(self):
        source_batch = self.print_tensor_with_shape(self.source_batch, 'source_batch')
        source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, self.source_identifier)

        target_batch = self.print_tensor_with_shape(self.target_batch, 'target_batch')
        target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        encoded_target = self.encoder.encode_inputs_to_vector(target_embbeding, self.target_identifier)

        batch_size = tf.shape(source_batch)[0]
        sentence_length = tf.shape(source_batch)[1]

        # reconstruction loss source
        reconstructed_source = self.decoder.decode_vector_to_sequence(encoded_source,
                                                                      self.decoder.get_zero_state(batch_size),
                                                                      source_embbeding, self.source_identifier)
        reconstructed_source_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
            reconstructed_source)
        source_reconstruction_loss = self.loss_handler.get_sentence_reconstruction_loss(source_batch,
                                                                                        reconstructed_source_logits)

        # reconstruction loss target
        reconstructed_target = self.decoder.decode_vector_to_sequence(encoded_target,
                                                                      self.decoder.get_zero_state(batch_size),
                                                                      target_embbeding, self.target_identifier)
        reconstructed_target_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
            reconstructed_target)
        target_reconstruction_loss = self.loss_handler.get_sentence_reconstruction_loss(target_batch,
                                                                                        reconstructed_target_logits)

        # professor forcing loss source
        professor_generated_source_embeddings = self.decoder.do_iterative_decoding(encoded_source,
                                                                                   self.source_identifier,
                                                                                   sentence_length)
        source_professor_loss = self.loss_handler.get_professor_forcing_loss(source_embbeding,
                                                                             professor_generated_source_embeddings)

        # professor forcing loss target
        professor_generated_target_embeddings = self.decoder.do_iterative_decoding(encoded_target,
                                                                                   self.target_identifier,
                                                                                   sentence_length)
        target_professor_loss = self.loss_handler.get_professor_forcing_loss(target_embbeding,
                                                                             professor_generated_target_embeddings)

        # semantic vector distance
        encoded_unstacked = tf.unstack(encoded_source)
        processed_encoded = []
        for e in encoded_unstacked:
            d = self.decoder.do_iterative_decoding(e, self.target_identifier, iterations_limit=-1)
            e_target = self.encoder.encode_inputs_to_vector(d, self.target_identifier)
            processed_encoded.append(e_target)
        encoded_again = tf.concat(processed_encoded, axis=0)
        semantic_distance_loss = self.loss_handler.get_context_vector_distance_loss(encoded_source, encoded_again)

        # anti discriminator loss
        discriminator_prediction_on_target = self.discriminator.encode_inputs_to_vector(target_embbeding)
        target_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_target, True)[1]
        decoded_fake_target = self.decoder.do_iterative_decoding(encoded_source, self.target_identifier,
                                                                 iterations_limit=sentence_length)
        discriminator_prediction_on_fake = self.discriminator.encode_inputs_to_vector(decoded_fake_target)
        fake_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_fake, False)[1]
        anti_discriminator_loss = target_loss + fake_loss

        total_loss = self.source_reconstruction_loss_coefficient * source_reconstruction_loss + \
                     self.target_reconstruction_loss_coefficient * target_reconstruction_loss + \
                     self.source_professor_loss_coefficient * source_professor_loss + \
                     self.target_professor_loss_coefficient * target_professor_loss + \
                     self.semantic_distance_loss_coefficient * semantic_distance_loss + \
                     self.anti_discriminator_loss_coefficient * anti_discriminator_loss

        var_list = self.encoder.get_trainable_parameters() + self.decoder.get_trainable_parameters() + \
                   self.embedding_translator.get_trainable_parameters()
        train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(total_loss,
                                                                                              var_list=var_list)
        return train_step, total_loss

