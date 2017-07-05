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

    def get_context_vector_distance_loss(self):
        pass

    def get_sentence_reconstruction_loss(self):
        pass

    def get_professor_forcing_loss(self):
        pass