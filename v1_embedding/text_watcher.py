import tensorflow as tf


class TextWatcher:
    def __init__(self, first_group_name, second_group_name):
        self.placeholder1, summary1 = self._create_assignable_scalar(first_group_name)
        self.placeholder2, summary2 = self._create_assignable_scalar(second_group_name)
        self.summary = tf.summary.merge([summary1, summary2])

    @staticmethod
    def _create_assignable_scalar(name):
        placeholder = tf.placeholder(dtype=tf.string, shape=(None), name='{}_placeholder'.format(name))
        summary = tf.summary.text(name, placeholder)
        return placeholder, summary
