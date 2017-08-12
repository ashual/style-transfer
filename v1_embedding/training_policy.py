class TrainingPolicy:
    def should_train_generator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        pass

    def should_train_discriminator(self, global_step, epoch_num, batch_index, prediction, new_loss, new_accuracy):
        pass

    def do_train_switch(self, start_training_generator=False):
        pass