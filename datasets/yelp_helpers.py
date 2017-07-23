import json
from random import shuffle

from datasets.dataset import Dataset


class YelpSentences(Dataset):
    def __init__(self, positive=True, limit_sentences=None):
        Dataset.__init__(self, limit_sentences=limit_sentences)
        if positive:
            with open('datasets/yelp/positive_reviews.json') as yelp:
                content = yelp.readlines()
        else:
            with open('datasets/yelp/negative_reviews.json') as yelp:
                content = yelp.readlines()
        self.content = [json.loads(s)['text'].lower() for s in content]
        shuffle(self.content)
