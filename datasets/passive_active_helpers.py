from datasets.dataset import Dataset


class PassiveActiveSentences(Dataset):
    def __init__(self, passive=True, limit_sentences=None, dataset_cache_dir=None, dataset_name=None):
        Dataset.__init__(self, limit_sentences=limit_sentences, dataset_cache_dir=dataset_cache_dir,
                         dataset_name=dataset_name)
        self.passive = passive

    def get_content_actual(self):
        if self.passive:
            with open('datasets/activepassive/passive.txt') as f:
                content = f.readlines()
        else:
            with open('datasets/activepassive/active.txt') as f:
                content = f.readlines()
        return content
