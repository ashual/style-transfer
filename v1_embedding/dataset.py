class Dataset:
    def __init__(self, limit_sentences=None):
        self.content = None
        self.limit_sentences = limit_sentences

    def get_content(self):
        return self.content if self.limit_sentences is None else self.content[:self.limit_sentences]
