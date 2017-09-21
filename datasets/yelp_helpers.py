from datasets.dataset import Dataset
from nltk import word_tokenize


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


def shuffle_sentence(sent):
    tokenize = word_tokenize(sent)
    return ' '.join(tokenize[:6])

if __name__ == "__main__":
    with open('datasets/yelp/pos.txt') as yelp:
        content = yelp.readlines()
    with open('datasets/yelp/pos_6.txt', 'w') as f:
        f.writelines("%s\n" % shuffle_sentence(l) for l in content)
