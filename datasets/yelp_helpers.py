import json
from random import shuffle


class YelpSentences:
    def __init__(self, positive=True):
        if positive:
            with open('datasets/yelp/positive_reviews.json') as yelp:
                content = yelp.readlines()
        else:
            with open('datasets/yelp/negative_reviews.json') as yelp:
                content = yelp.readlines()
        shuffle(content)
        self.content_iterator = iter(content)

    def __iter__(self):
        return self

    def __next__(self):
        return json.loads(next(self.content_iterator))['text']
