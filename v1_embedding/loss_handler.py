import tensorflow as tf
from v1_embedding.base_model import BaseModel
from tensorflow.contrib.losses import sigmoid_cross_entropy


class LossHandler(BaseModel):

    def __init__(self, vocabulary_length):
        BaseModel.__init__(self)
        self.vocabulary_length = vocabulary_length

    def get_context_vector_distance_loss(self, encoded_source, encoded_dest):
        squared_difference = tf.squared_difference(encoded_source, encoded_dest)
        return tf.reduce_mean(squared_difference)

    def get_sentence_reconstruction_loss(self, labels, logits):
        # get the places without padding
        padding_mask = tf.not_equal(labels, self.vocabulary_length)
        # zero out places with padding, required for the softmax to be valid
        non_padding_labels = tf.where(tf.logical_not(padding_mask), tf.zeros_like(labels), labels)
        # get the cross entropy in each place
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=non_padding_labels, logits=logits)
        # cast the padding
        casted_padding_mask = tf.cast(padding_mask, tf.float32)
        # remove the places with padding
        cross_entropy_no_padding = tf.multiply(cross_entropy, casted_padding_mask)
        # get the mean with respect to the non padded elements only
        cross_entropy_sum = tf.reduce_sum(cross_entropy_no_padding)
        return tf.divide(cross_entropy_sum, tf.reduce_sum(casted_padding_mask))

    def get_accuracy(self, labels, prediction):
        # get the places without padding
        padding_mask = tf.not_equal(labels, self.vocabulary_length)
        # get the places where label == prediction that are not padding
        correct_prediction = tf.equal(labels, prediction)
        relevant_correct_predictions = tf.cast(tf.logical_and(padding_mask, correct_prediction), tf.float32)
        # cast the padding
        casted_padding_mask = tf.cast(padding_mask, tf.float32)
        return tf.divide(tf.reduce_sum(relevant_correct_predictions), tf.reduce_sum(casted_padding_mask))

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
