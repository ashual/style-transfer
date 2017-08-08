import tensorflow as tf


class BaseModel:
    def __init__(self, name=None):
        self.should_print = tf.placeholder_with_default(False, shape=())
        self.name = self.__class__.__name__
        if name:
            self.name = '{}_{}'.format(name, self.name)
        self.trainable_parameters = None
        print('{} created'.format(self.name))

    def print_tensor_with_shape(self, tensor, name):
        return tensor
        # Print is CPU based, removing it for now
        # return tf.cond(self.should_print,
        #                lambda: tf.Print(
        #                    tf.Print(tensor, [tensor], message=name + ":"),
        #                    [tf.shape(tensor)], message=name + " shape:"),
        #                lambda: tf.identity(tensor))

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

    def concat_identifier(self, inputs, identifier):
        if identifier is None:
            identified_inputs = inputs
        else:
            batch_size = tf.shape(inputs)[0]
            sentence_length = tf.shape(inputs)[1]

            # the input sequence s.t (batch, time, embedding)
            inputs = self.print_tensor_with_shape(inputs, "inputs")

            # create the input: (batch, time, embedding; domain)
            identifier = self.print_tensor_with_shape(identifier, "domain_identifier")
            identifier_tiled = identifier * tf.ones([batch_size, sentence_length, 1])
            identified_inputs = tf.concat((inputs, identifier_tiled), axis=2)
        return identified_inputs
