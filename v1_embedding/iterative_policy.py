import tensorflow as tf


class IterativePolicy:
    def __init__(self, start_training_generator, generator_steps=100, discriminator_steps=100):
        self.train_generator = tf.Variable(start_training_generator, trainable=False, dtype=tf.bool)
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.generator_steps = generator_steps
        self.discriminator_steps = discriminator_steps

    def should_train_generator(self):
        return self.train_generator

    def notify(self):
        # if we are training the generator and there are no steps left
        switch_generator = tf.logical_and(self.train_generator, tf.equal(self.counter, self.generator_steps))
        # if we are training the discriminator and there are no steps left
        switch_discriminator = tf.logical_and(tf.logical_not(self.train_generator), tf.equal(self.counter,
                                                                                             self.discriminator_steps))
        switch_component = tf.logical_or(switch_generator, switch_discriminator)
        return tf.cond(
            switch_component,
            # if we need to switch, negate train_generator and assign 0 to counter
            lambda: tf.group(
                tf.assign(self.train_generator, tf.logical_not(self.train_generator)),
                tf.assign(self.counter, 0)
            ),
            # otherwise, just raise the counter
            lambda: tf.group(
                tf.assign_add(self.counter, 1)
            )
        )
