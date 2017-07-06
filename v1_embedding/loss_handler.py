import tensorflow as tf
from v1_embedding.base_model import BaseModel


class LossHandler(BaseModel):

    def __init__(self, embedding_translator, encoder, decoder, discriminator):
        self.embedding_translator = embedding_translator
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def get_w_loss(self):
        pass

    def get_context_vector_distance_loss(self, encoded_source, encoded_dest):
        squared_difference = tf.squared_difference(encoded_source, encoded_dest)
        return tf.reduce_mean(squared_difference)

    def get_sentence_reconstruction_loss(self, source_logits, dest_logits):
        cross_entropy = source_logits * tf.log(dest_logits)
        return tf.reduce_mean(cross_entropy)

    def get_professor_forcing_loss(self, inputs, generated_inputs):
        squared_difference = tf.squared_difference(inputs, generated_inputs)
        return tf.reduce_mean(squared_difference)