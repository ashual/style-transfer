from nltk import word_tokenize


class Dataset:
    def __init__(self, limit_sentences=None):
        self.content = None
        self.limit_sentences = limit_sentences

    def get_content(self):
        return self.content if self.limit_sentences is None else self.content[:self.limit_sentences]

    def get_word_dictionary(self):
        content = self.get_content()
        word_dict = dict()
        for sentence in content:
            for word in word_tokenize(sentence):
                word_lower = word.lower()
                if word_lower not in word_dict:
                    word_dict[word_lower] = True
        return word_dict
