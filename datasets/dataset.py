from nltk import word_tokenize
import os
from random import shuffle


class Dataset:
    def __init__(self, limit_sentences, validation_limit_sentences, dataset_cache_dir=None, dataset_name=None):
        self.content = None
        self.validation_content = None
        self.limit_sentences = limit_sentences
        self.validation_limit_sentences = validation_limit_sentences
        self.dataset_cache_dir = dataset_cache_dir
        self.dataset_cache_file = None
        self.validation_dataset_cache_file = None
        if dataset_cache_dir is not None:
            if not os.path.exists(dataset_cache_dir):
                os.makedirs(dataset_cache_dir)
            self.dataset_cache_file = os.path.join(dataset_cache_dir,
                                                   'dataset.txt' if dataset_name is None else dataset_name + '.txt')
            self.validation_dataset_cache_file = os.path.join(dataset_cache_dir,
                                                              'validation_dataset.txt' if dataset_name is None else
                                                              'validation_' + dataset_name + '.txt')

    def get_content(self):
        if self.content is None:
            if self.dataset_cache_file is not None and os.path.exists(self.dataset_cache_file):
                with open(self.dataset_cache_file) as f:
                    self.content = f.readlines()
                with open(self.validation_dataset_cache_file) as f:
                    self.validation_content = f.readlines()
            else:
                full_content = self.get_content_actual()
                if len(full_content) < self.limit_sentences + self.validation_limit_sentences:
                    raise Exception('there are no enough sentences in the dataset')
                shuffle(full_content)
                self.content = full_content[:self.limit_sentences]
                self.validation_content = full_content[
                                          self.limit_sentences:self.limit_sentences + self.validation_limit_sentences]
                if self.dataset_cache_file is not None:
                    with open(self.dataset_cache_file, 'w') as f:
                        if self.content[0][-1] == '\n':
                            f.writelines("%s" % l for l in self.content)
                        else:
                            f.writelines("%s\n" % l for l in self.content)
                if self.validation_dataset_cache_file is not None:
                    with open(self.validation_dataset_cache_file, 'w') as f:
                        if self.validation_content[0][-1] == '\n':
                            f.writelines("%s" % l for l in self.validation_content)
                        else:
                            f.writelines("%s\n" % l for l in self.validation_content)
        return self.content, self.validation_content

    def get_content_actual(self):
        pass

    def get_word_dictionary(self):
        content, validation_content = self.get_content()
        word_dict = dict()
        for sentence in content + validation_content:
            for word in word_tokenize(sentence):
                word_lower = word.lower()
                if word_lower not in word_dict:
                    word_dict[word_lower] = True
        return word_dict
