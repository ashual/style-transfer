from random import shuffle
from datasets.basic_dataset import BasicDataset
from datasets.batch_iterator import BatchIterator


class MultiBatchIterator:
    def __init__(self, datasets, embedding_handler, sentence_len, batch_size):
        self.datasets = datasets
        self.min_content_length = min([len(d.get_content()) for d in self.datasets])
        self.embedding_handler = embedding_handler
        self.sentence_len = sentence_len
        self.batch_size = batch_size

    def get_iterator(self, dataset):
        content = dataset.get_content()
        # since we want the data in each epoch to be different we shuffle beforehand
        shuffle(content)
        # take a random prefix which is the size of the smallest dataset
        content = content[:self.min_content_length]
        basic_dataset = BasicDataset(content)
        batch_iterator = BatchIterator(basic_dataset, self.embedding_handler, self.sentence_len, self.batch_size)
        return batch_iterator

    def __iter__(self):
        for res in zip(*[self.get_iterator(d) for d in self.datasets]):
            yield res
