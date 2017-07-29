import json
from random import shuffle

from datasets.dataset import Dataset


class YelpSentences(Dataset):
    def __init__(self, positive=True, limit_sentences=None, dataset_cache_dir=None, dataset_name=None):
        Dataset.__init__(self, limit_sentences=limit_sentences, dataset_cache_dir=dataset_cache_dir,
                         dataset_name=dataset_name)
        self.positive = positive

    def get_content_actual(self):
        if self.positive:
            with open('datasets/yelp/positive_reviews.json') as yelp:
                content = yelp.readlines()
        else:
            with open('datasets/yelp/negative_reviews.json') as yelp:
                content = yelp.readlines()
        content = [json.loads(s)['text'].lower() for s in content]
        shuffle(content)
        return content