import tensorflow as tf


class TfCounter:
    def __init__(self, name):
        self.count = tf.Variable(0, dtype=tf.int32, trainable=False, name="{}_counter".format(name))
        self.update = tf.assign_add(self.count, 1)
