from random import shuffle
from datasets.batch_iterator import BatchIterator


class MultiBatchIterator:
    def __init__(self, contents, embedding_handler, sentence_len, batch_size):
        self.contents = contents
        self.min_content_length = min([len(d) for d in self.contents])
        self.embedding_handler = embedding_handler
        self.sentence_len = sentence_len
        self.batch_size = batch_size

    def get_iterator(self, content):
        # since we want the data in each epoch to be different we shuffle beforehand
        shuffle(content)
        # take a random prefix which is the size of the smallest dataset
        content = content[:self.min_content_length]
        batch_iterator = BatchIterator(content, self.embedding_handler, self.sentence_len, self.batch_size)
        return batch_iterator

    def __iter__(self):
        for res in zip(*[self.get_iterator(d) for d in self.contents]):
            yield res

    @staticmethod
    def preprocess(datasets):
        contents = []
        validation_contents = []
        for dataset in datasets:
            content, validation_content = dataset.get_content()
            contents.append(content)
            validation_contents.append(validation_content)
        return contents, validation_contents
