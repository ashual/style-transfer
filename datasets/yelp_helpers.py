from datasets.dataset import Dataset


class YelpSentences(Dataset):
    def __init__(self, positive=True, limit_sentences=None, dataset_cache_dir=None, dataset_name=None):
        Dataset.__init__(self, limit_sentences=limit_sentences, dataset_cache_dir=dataset_cache_dir,
                         dataset_name=dataset_name)
        self.positive = positive

    def get_content_actual(self):
        if self.positive:
            with open('datasets/yelp/pos.txt') as yelp:
                content = yelp.readlines()
        else:
            with open('datasets/yelp/neg.txt') as yelp:
                content = yelp.readlines()
        return content
