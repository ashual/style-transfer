import tensorflow as tf


class BaseModel:
    def __init__(self):
        self.should_print = tf.placeholder_with_default(False, shape=())

    def print_tensor_with_shape(self, tensor, name):
        return tf.cond(self.should_print,
                       lambda: tf.Print(
                           tf.Print(tensor, [tensor], message=name + ":"),
                           [tf.shape(tensor)], message=name + " shape:"),
                       lambda: tf.identity(tensor))
