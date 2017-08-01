import tensorflow as tf
import os


class SaverWrapper:
    def __init__(self, saver_dir, model_name):
        self.saver_dir = saver_dir
        self.saver_path = os.path.join(self.saver_dir, model_name)
        self.saver = tf.train.Saver()
        if not os.path.exists(self.saver_dir):
                os.makedirs(self.saver_dir)
        print('models are saved to: {}'.format(self.saver_dir))
        print()

    def load_model(self, sess):
        checkpoint_path = tf.train.get_checkpoint_state(self.saver_dir)
        if checkpoint_path is not None:
            self.saver.restore(sess, checkpoint_path.model_checkpoint_path)
            print('Model restored from file: {}'.format(checkpoint_path.model_checkpoint_path))
        else:
            print('Model not found in: {}'.format(self.saver_dir))
        print()

    def save_model(self, sess, save_retries=3):
        for i in range(save_retries):
            try:
                # save model
                self.saver.save(sess, self.saver_path)
                print('Model saved\n')
                return True
            except:
                pass
        print('Failed to save model\n')
        print()
        return False