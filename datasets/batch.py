class Batch:
    def __init__(self):
        self.sentences = []
        self.lengths = []

    def add(self, sentence, length):
        self.sentences.append(sentence)
        self.lengths.append(length)

    def get_len(self):
        return len(self.sentences)
