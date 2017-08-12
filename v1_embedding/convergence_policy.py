from v1_embedding.training_policy import TrainingPolicy


class ConvergencePolicy(TrainingPolicy):
    def __init__(self, minimal_steps=1):
        self.train_generator = False
        # privates
        self.history_weight = 2.0 / 3.0
        self.running_loss = None
        self.steps_counter = 0
        self.minimal_steps = minimal_steps

    def update_running_accuracy(self, new_loss):
        if self.running_loss is None:
            self.running_loss = new_loss
        else:
            self.running_loss = self.history_weight * self.running_loss + (1 - self.history_weight) * new_loss

    def should_train_generator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        self.update_running_accuracy(new_loss)
        result = self.steps_counter < self.minimal_steps or new_loss > self.running_loss
        self.steps_counter += 1
        return result

    def should_train_discriminator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        self.update_running_accuracy(new_loss)
        result = self.steps_counter < self.minimal_steps or new_loss < self.running_loss
        self.steps_counter += 1
        return result

    def do_train_switch(self, start_training_generator=False):
        self.train_generator = start_training_generator
        self.running_loss = None
        self.steps_counter = 0
