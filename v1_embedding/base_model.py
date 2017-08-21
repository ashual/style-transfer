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

    @staticmethod
    def create_summaries(v):
        return [tf.summary.scalar(v.name + '_mean', tf.reduce_mean(v)),
                tf.summary.scalar(v.name + '_l2_norm', tf.norm(v)),
                tf.summary.scalar(v.name + '_max_norm', tf.reduce_max(tf.abs(v))),
                tf.summary.histogram(v.name, v)]

    def get_trainable_parameters_summaries(self):
        res = []
        for v in self.get_trainable_parameters():
            res += BaseModel.create_summaries(v)
        return res

    @staticmethod
    def concat_identifier(inputs, identifier):
        if identifier is None:
            # result is (batch, time, embedding)
            identified_inputs = inputs
        else:
            # result is (batch, time, embedding; domain)
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]
            identifier_tiled = identifier * tf.ones([batch_size, sentence_length, 1])
            identified_inputs = tf.concat((inputs, identifier_tiled), axis=2)
        return identified_inputs
