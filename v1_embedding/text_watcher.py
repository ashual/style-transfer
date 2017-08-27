import tensorflow as tf


class TextWatcher:
    def __init__(self, names):
        self.placeholders = {}
        summaries = []
        for name in names:
            placeholder, summary = self._create_assignable_scalar(name)
            self.placeholders[name] = placeholder
            summaries.append(summary)
        self.summary = tf.summary.merge(summaries)

    @staticmethod
    def _create_assignable_scalar(name):
        placeholder = tf.placeholder(dtype=tf.string, shape=(None), name='{}_placeholder'.format(name))
        summary = tf.summary.text(name, placeholder)
        return placeholder, summary
