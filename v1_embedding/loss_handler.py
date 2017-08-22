import tensorflow as tf
from v1_embedding.base_model import BaseModel


class LossHandler(BaseModel):

    def __init__(self, vocabulary_length):
        BaseModel.__init__(self)
        self.vocabulary_length = vocabulary_length
        self.epsilon = 0.001

    def get_context_vector_distance_loss(self, encoded_source, encoded_dest):
        with tf.variable_scope('ContextVectorDistanceLoss'):
            squared_difference = tf.squared_difference(encoded_source, encoded_dest)
            return tf.reduce_mean(squared_difference)

    def get_distance_loss(self, source, target, padding_mask):
        sq_difference = tf.sqrt(tf.reduce_sum(tf.squared_difference(source, target), axis=-1))
        mask = tf.where(padding_mask, tf.ones_like(padding_mask, dtype=tf.float32),
                        tf.zeros_like(padding_mask, dtype=tf.float32))
        sum = tf.reduce_sum(sq_difference * mask)
        mask_sum = tf.reduce_sum(mask)
        return sum / mask_sum

    def get_margin_loss(self, target, padding_mask, random_words, margin):
        len_random_words = tf.shape(random_words)[2]
        target_expand = tf.expand_dims(target, 2)
        target_expand = tf.tile(target_expand, [1, 1, len_random_words, 1])

        per_random_word_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(target_expand, random_words), axis=-1))
        per_word_margin_loss = tf.maximum(0.0, margin-per_random_word_distance)
        per_word_distance = tf.reduce_mean(per_word_margin_loss, axis=-1)

        mask = tf.where(padding_mask, tf.ones_like(padding_mask, dtype=tf.float32),
                        tf.zeros_like(padding_mask, dtype=tf.float32))
        sum = tf.reduce_sum(per_word_distance * mask)
        mask_sum = tf.reduce_sum(mask)
        return sum / mask_sum

    def get_margin_loss_v2(self, true_embeddings, decoded_embeddings, random_words_embeddings, padding_mask, margin):
        len_random_words = tf.shape(random_words_embeddings)[2]

        sq_difference = tf.sqrt(tf.reduce_sum(tf.squared_difference(true_embeddings, decoded_embeddings), axis=-1))
        sq_difference_expand = tf.expand_dims(sq_difference, 2)
        sq_difference_expand = tf.tile(sq_difference_expand, [1, 1, len_random_words])

        target_expand = tf.expand_dims(decoded_embeddings, 2)
        target_expand = tf.tile(target_expand, [1, 1, len_random_words, 1])
        per_random_word_distance = tf.sqrt(
        tf.reduce_sum(tf.squared_difference(target_expand, random_words_embeddings), axis=-1))

        per_word_margin_loss = tf.maximum(0.0, margin + sq_difference_expand - per_random_word_distance)
        per_word_distance = tf.reduce_mean(per_word_margin_loss, axis=-1)

        mask = tf.where(padding_mask, tf.ones_like(padding_mask, dtype=tf.float32),
                        tf.zeros_like(padding_mask, dtype=tf.float32))
        sum = tf.reduce_sum(per_word_distance * mask)
        mask_sum = tf.reduce_sum(mask)

        return sum / mask_sum

    def get_sentence_reconstruction_loss(self, labels, logits, padding_mask=None):
        with tf.variable_scope('SentenceReconstructionLoss'):
            # get the places without padding
            if padding_mask is None:
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

    def _stable_log(self, input):
        with tf.variable_scope('StableLog'):
            input = tf.maximum(input, self.epsilon)
            input = tf.minimum(input, 1.0 - self.epsilon)
            return tf.log(input)

    def get_discriminator_loss(self, prediction_transferred, prediction_target):
        with tf.variable_scope('DiscriminatorLoss'):
            transferred_accuracy = tf.reduce_mean(tf.cast(tf.less(prediction_transferred, 0.5), tf.float32))
            transferred_loss = -tf.reduce_mean(self._stable_log(1.0 - prediction_transferred))

            target_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(prediction_target, 0.5), tf.float32))
            target_loss = -tf.reduce_mean(self._stable_log(prediction_target))

            # total loss is the sum of losses
            total_loss = transferred_loss + target_loss
            # total accuracy is the avg of accuracies
            total_accuracy = 0.5 * (transferred_accuracy + target_accuracy)
            return total_loss, total_accuracy

    def get_accuracy(self, labels, prediction):
        with tf.variable_scope('ComputeAccuracy'):
            # get the places without padding
            padding_mask = tf.not_equal(labels, self.vocabulary_length)
            # get the places where label == prediction that are not padding
            correct_prediction = tf.equal(labels, prediction)
            relevant_correct_predictions = tf.cast(tf.logical_and(padding_mask, correct_prediction), tf.float32)
            # cast the padding
            casted_padding_mask = tf.cast(padding_mask, tf.float32)
            return tf.divide(tf.reduce_sum(relevant_correct_predictions), tf.reduce_sum(casted_padding_mask))
