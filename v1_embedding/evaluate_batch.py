import datetime
import yaml
import tensorflow as tf

from v1_embedding.model_trainer import ModelTrainer
from v1_embedding.logger import init_logger
from datasets.batch_iterator import BatchIterator

if __name__ == "__main__":
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("config/gan.yml", 'r') as ymlfile:
        config = yaml.load(ymlfile)
    with open("config/operational.yml", 'r') as ymlfile:
        operational_config = yaml.load(ymlfile)
    init_logger(name)
    print('------------ Config ------------')
    print(yaml.dump(config))
    print('------------ Operational Config ------------')
    print(yaml.dump(operational_config))
    model_trainer = ModelTrainer(config, operational_config)
    model = model_trainer.model

    # read input file as list of sentences
    with open('input.txt') as f:
        content = f.readlines()

    batch_iterator = BatchIterator(content, model.embedding_handler, config['sentence']['min_length'], 100)
    with tf.Session() as sess:
        model_trainer.saver_wrapper.load_model(sess)
        for b in batch_iterator:
            new_batch = [b, b] # fake multi batch
            _, _, original_source, transferred = model_trainer.transfer_batch(sess, new_batch)
            for i in range(len(original_source)):
                print('original_source: {}'.format(original_source[i]))
                print('transferred: {}'.format(transferred[i]))
            break


