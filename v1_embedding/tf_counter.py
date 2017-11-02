import tensorflow as tf


class TfCounter:
    def __init__(self, name):
        self.count = tf.Variable(0, dtype=tf.int32, trainable=False, name="{}_counter".format(name))
        self.update = tf.assign_add(self.count, 1)

    def increase_if(self, condition):
        constant = tf.cond(
            pred=condition,
            true_fn=lambda: tf.constant(1),
            false_fn=lambda: tf.constant(0)
        )
        return tf.assign_add(self.count, constant)
