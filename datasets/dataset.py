from nltk import word_tokenize
import os
from random import shuffle


class Dataset:
    def __init__(self, limit_sentences=None, dataset_cache_dir=None, dataset_name=None):
        self.content = None
        self.limit_sentences = limit_sentences
        self.dataset_cache_dir = dataset_cache_dir
        self.dataset_cache_file = None
        if dataset_cache_dir is not None:
            if not os.path.exists(dataset_cache_dir):
                os.makedirs(dataset_cache_dir)
            self.dataset_cache_file = os.path.join(dataset_cache_dir,
                                                   'dataset.txt' if dataset_name is None else dataset_name + '.txt')

    def get_content(self):
        if self.content is None:
            if self.dataset_cache_file is not None and os.path.exists(self.dataset_cache_file):
                with open(self.dataset_cache_file) as f:
                    self.content = f.readlines()
            else:
                self.content = self.get_content_actual()
                shuffle(self.content)
                if self.limit_sentences is not None:
                    self.content = self.content[:self.limit_sentences]
                if self.dataset_cache_file is not None:
                    with open(self.dataset_cache_file, 'w') as f:
                        if self.content[0][-1] == '\n':
                            f.writelines("%s" % l for l in self.content)
                        else:
                            f.writelines("%s\n" % l for l in self.content)
        return self.content

    def get_content_actual(self):
        pass

    def get_word_dictionary(self):
        content = self.get_content()
        word_dict = dict()
        for sentence in content:
            for word in word_tokenize(sentence):
                word_lower = word.lower()
                if word_lower not in word_dict:
                    word_dict[word_lower] = True
        return word_dict
