import random
from datasets.dataset import Dataset
from nltk import word_tokenize


class Scramble(Dataset):
    def __init__(self, is_scramble, limit_sentences, dataset_cache_dir, dataset_name):
        Dataset.__init__(self, limit_sentences, dataset_cache_dir, dataset_name)
        self.is_scramble = is_scramble

    @staticmethod
    def shuffle_sentence(sent):
        tokenize = word_tokenize(sent)
        random.shuffle(tokenize)
        return ' '.join(tokenize)

    def get_content_actual(self):
        with open('datasets/yelp/pos.txt') as yelp:
            content = yelp.readlines()
        if self.is_scramble:
            new_content = []
            for sent in content:
                new_content.append(Scramble.shuffle_sentence(sent))
            return new_content
        else:
            return content
