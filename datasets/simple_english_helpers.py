from datasets.dataset import Dataset


class SimpleEnglishSentences(Dataset):
    def __init__(self, regular=True, limit_sentences=None, dataset_cache_dir=None, dataset_name=None):
        Dataset.__init__(self, limit_sentences=limit_sentences, dataset_cache_dir=dataset_cache_dir,
                         dataset_name=dataset_name)
        self.regular = regular

    def get_content_actual(self):
        if self.regular:
            with open('datasets/simple-english/regular.txt') as simple_english:
                content = simple_english.readlines()
        else:
            with open('datasets/simple-english/simple.txt') as simple_english:
                content = simple_english.readlines()
        return content
