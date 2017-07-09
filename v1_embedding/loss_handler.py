import tensorflow as tf
from v1_embedding.base_model import BaseModel
from tensorflow.contrib.losses import sigmoid_cross_entropy


class LossHandler(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)

    def get_context_vector_distance_loss(self, encoded_source, encoded_dest):
        squared_difference = tf.squared_difference(encoded_source, encoded_dest)
        return tf.reduce_mean(squared_difference)

    def get_sentence_reconstruction_loss(self, source_logits, dest_logits):
        cross_entropy = source_logits * tf.log(dest_logits)
        return tf.reduce_mean(cross_entropy)

    def get_professor_forcing_loss(self, inputs, generated_inputs):
        squared_difference = tf.squared_difference(inputs, generated_inputs)
        return tf.reduce_mean(squared_difference)

    def get_discriminator_loss(self, logits, is_real_images):
        # TODO - need to fix to w-loss (best to change to a 2 elements logits
        if is_real_images:
            d_loss_target = tf.zeros_like(logits)
            g_loss_target = tf.ones_like(logits)
        else:
            d_loss_target = tf.ones_like(logits)
            g_loss_target = tf.zeros_like(logits)
        d_loss = sigmoid_cross_entropy(logits, d_loss_target)
        g_loss = sigmoid_cross_entropy(logits, g_loss_target)
        return d_loss, g_loss
