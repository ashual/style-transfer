import json



class YelpSentences:
    def __init__(self, positive=True):
        if positive:
            with open('datasets/yelp/positive_reviews.json') as yelp:
                content = yelp.readlines()
        else:
            with open('datasets/yelp/negative_reviews.json') as yelp:
                content = yelp.readlines()
        self.content = [json.loads(s)['text'] for s in content]

