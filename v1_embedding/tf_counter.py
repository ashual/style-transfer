import tensorflow as tf


class TfCounter:
    def __init__(self, name):
        self.count = tf.Variable(0, dtype=tf.int32, trainable=False, name="{}_counter".format(name))
        new_value = tf.add(self.count, tf.constant(1))
        self.update = tf.assign(self.count, new_value)