from datasets.dataset import Dataset


class BasicDataset(Dataset):
    def __init__(self, content):
        Dataset.__init__(self, limit_sentences=None, dataset_cache_dir=None, dataset_name=None)
        self.c = content

    def get_content_actual(self):
        return self.c
