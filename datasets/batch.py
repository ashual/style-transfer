class Batch:
    def __init__(self):
        self.sentences = []
        self.lengths = []

    def add(self, sentences, lengths):
        self.sentences.append(sentences)
        self.lengths.append(lengths)

    def get_len(self):
        return len(self.sentences)
