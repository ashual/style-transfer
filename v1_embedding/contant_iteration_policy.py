from v1_embedding.training_policy import TrainingPolicy


class ConstantIterationPolicy(TrainingPolicy):
    def __init__(self, generator_steps=100, discriminator_steps=100):
        self.train_generator = False
        self.counter = 0
        self.generator_steps = generator_steps
        self.discriminator_steps = discriminator_steps

    def should_train_generator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        res = self.counter < self.generator_steps
        self.counter += 1
        return res

    def should_train_discriminator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        res = self.counter < self.discriminator_steps
        self.counter += 1
        return res

    def do_train_switch(self, start_training_generator=False):
        self.train_generator = start_training_generator
        self.counter = 0
