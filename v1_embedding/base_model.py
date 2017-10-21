import tensorflow as tf


class BaseModel:
    def __init__(self, name=None):
        self.name = self.__class__.__name__
        if name:
            self.name = '{}_{}'.format(name, self.name)
        self.trainable_parameters = None
        print('{} created'.format(self.name))

    @staticmethod
    def create_input_parameters(input_size, output_size):
        w = tf.Variable(tf.random_normal(shape=(input_size, output_size)), dtype=tf.float32)
        b = tf.Variable(tf.random_normal(shape=(output_size,)), dtype=tf.float32)
        return w, b

    def get_trainable_parameters(self):
        if self.trainable_parameters is None:
            self.trainable_parameters = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        return self.trainable_parameters

