import tensorflow as tf

from datasets.yelp_helpers import YelpSentences
from v1_embedding.embedding_translator import EmbeddingTranslator
from v1_embedding.embedding_encoder import EmbeddingEncoder
from v1_embedding.embedding_decoder import EmbeddingDecoder
from v1_embedding.embedding_discriminator import EmbeddingDiscriminator
from v1_embedding.model_trainer_base import ModelTrainerBase
from v1_embedding.word_indexing_embedding_handler import WordIndexingEmbeddingHandler


class ModelTrainer(ModelTrainerBase):
    def __init__(self, config_file, operational_config_file):
        ModelTrainerBase.__init__(self, config_file=config_file, operational_config_file=operational_config_file)

        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        self.discriminator_dropout_placeholder = tf.placeholder(tf.float32, shape=())
        # placeholder for source sentences (batch, time)=> index of word
        self.source_batch = tf.placeholder(tf.int64, shape=(None, None))
        # placeholder for source sentences (batch, time)=> index of word
        self.target_batch = tf.placeholder(tf.int64, shape=(None, None))

        self.source_identifier = tf.ones(shape=())
        self.target_identifier = -1 * tf.ones(shape=())

        self.dataset_neg = YelpSentences(positive=False, limit_sentences=self.config['limit_sentences'],
                                         dataset_cache_dir=self.dataset_cache_dir, dataset_name='neg')
        self.dataset_pos = YelpSentences(positive=True, limit_sentences=self.config['limit_sentences'],
                                         dataset_cache_dir=self.dataset_cache_dir, dataset_name='pos')
        self.embedding_handler = WordIndexingEmbeddingHandler(
            self.embedding_dir,
            [self.dataset_neg, self.dataset_pos],
            self.config['word_embedding_size'],
            self.config['min_word_occurrences']
        )
        self.embedding_translator = EmbeddingTranslator(self.embedding_handler,
                                                        self.config['model']['translation_hidden_size'],
                                                        self.config['train_embeddings'])
        self.encoder = EmbeddingEncoder(self.config['encoder_hidden_states'], self.dropout_placeholder,
                                        self.config['bidirectional_encoder'])
        self.decoder = EmbeddingDecoder(self.embedding_handler.get_embedding_size(),
                                        self.config['decoder_hidden_states'],
                                        self.embedding_translator, self.dropout_placeholder)
        self.target_discriminator = EmbeddingDiscriminator(self.config['discriminator_hidden_states'],
                                                           self.config['model']['translation_hidden_size'],
                                                           self.discriminator_dropout_placeholder,
                                                           self.config['bidirectional_discriminator'])

    def _get_discriminator_loss_from_encoded(self, encoded_source, encoded_target, target_embbeding, target_identifier,
                                             target_descriminator, sentence_length):
        teacher_forced_target = self.decoder.do_teacher_forcing(encoded_target, target_embbeding[:, :-1, :],
                                                                target_identifier)
        loss_true = -tf.reduce_mean(tf.log(target_descriminator.predict(teacher_forced_target)))
        fake_targets = self.decoder.do_iterative_decoding(encoded_source, target_identifier, sentence_length-1)

        loss_fake = -tf.reduce_mean(tf.log(1.0 - target_descriminator.predict(fake_targets)))
        return loss_true + loss_fake

    def get_discriminator_loss(self, source_batch, target_batch, source_identifier, target_identifier,
                               target_descriminator):
        sentence_length = tf.shape(source_batch)[1]
        target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        encoded_target = self.encoder.encode_inputs_to_vector(target_embbeding, target_identifier)

        source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, source_identifier)

        return self._get_discriminator_loss_from_encoded(encoded_source, encoded_target, target_embbeding,
                                                         target_identifier, target_descriminator ,sentence_length)


        # target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        # discriminator_prediction_on_target = self.discriminator.predict(target_embbeding)
        # target_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_target, True)[0]
        #
        # source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        # encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, self.source_identifier)
        # sentence_length = tf.shape(target_batch)[1]
        # decoded_fake_target = self.decoder.do_iterative_decoding(encoded_source, self.target_identifier,
        #                                                          iterations_limit=sentence_length)
        # discriminator_prediction_on_fake = self.discriminator.predict(decoded_fake_target)
        # fake_loss = self.loss_handler.get_discriminator_loss(discriminator_prediction_on_fake, False)[0]
        #
        # total_loss = target_loss + fake_loss
        # var_list = self.discriminator.get_trainable_parameters()
        # train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(total_loss,
        #                                                                                       var_list=var_list)
        # return train_step, total_loss

    def get_generator_loss(self, source_batch, target_batch, source_identifier, target_identifier, target_descriminator):
        source_embbeding = self.embedding_translator.embed_inputs(source_batch)
        encoded_source = self.encoder.encode_inputs_to_vector(source_embbeding, source_identifier)

        target_embbeding = self.embedding_translator.embed_inputs(target_batch)
        encoded_target = self.encoder.encode_inputs_to_vector(target_embbeding, target_identifier)

        sentence_length = tf.shape(source_batch)[1]

        # reconstruction loss
        source_decoded_to_source = self.decoder.do_teacher_forcing(encoded_source, source_embbeding[:, :-1, :],
                                                                   source_identifier)
        reconstructed_source_logits = self.embedding_translator.translate_embedding_to_vocabulary_logits(
            source_decoded_to_source)
        reconstruction_loss = tf.reduce_mean(
            tf.squared_difference(
                tf.one_hot(source_batch, tf.shape(reconstructed_source_logits)[-1]),
                reconstructed_source_logits)
        )

        # semantic vector distance
        encoded_unstacked = tf.unstack(encoded_source)
        processed_encoded = []
        for e in encoded_unstacked:
            d = self.decoder.do_iterative_decoding(e, target_identifier, iterations_limit=-1)
            e_target = self.encoder.encode_inputs_to_vector(d, target_identifier)
            processed_encoded.append(e_target)
        encoded_again = tf.concat(processed_encoded, axis=0)
        semantic_distance_loss = tf.reduce_mean(tf.squared_difference(encoded_source, encoded_again))

        # professor forcing loss source
        anti_d_loss = -self._get_discriminator_loss_from_encoded(encoded_source, encoded_target, target_embbeding,
                                                                 target_identifier, target_descriminator,
                                                                 sentence_length)
        return self.config['reconstruction_coefficient'] * reconstruction_loss \
               + self.config['semantic_distance_coefficient'] * semantic_distance_loss \
               + anti_d_loss
        # teacher_forced_target = self.decoder.do_teacher_forcing(encoded_target, target_embbeding[:, :-1, :],
        #                                                         target_identifier)
        #
        #
        #
        # var_list = self.encoder.get_trainable_parameters() + self.decoder.get_trainable_parameters() + \
        #            self.embedding_translator.get_trainable_parameters()
        # train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(total_loss,
        #                                                                                       var_list=var_list)
        # for v in tf.trainable_variables():
        #     print(v.name)
        # return train_step, total_loss

