from datasets.bible_helpers import Bibles
import datasets.yelp_helpers as yelp
from nltk import word_tokenize


class BatchIterator:
    def __init__(self, dataset, embedding_handler, sentence_len, batch_size, limit_sentences=None):
        self.embedding_handler = embedding_handler
        if dataset == 'bible':
            # TODO: decide how to return both params
            self.text_iterator = Bibles('a', 'b')
        elif dataset == 'yelp_positive':
            self.text_iterator = yelp.YelpSentences(positive=True)
        elif dataset == 'yelp_negative':
            self.text_iterator = yelp.YelpSentences(positive=False)
        self.limit_sentences = limit_sentences
        self.counter = 0
        self.sentence_len = sentence_len
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.limit_sentences and self.counter >= self.limit_sentences:
            raise StopIteration

        sentences = []
        for sentence in self.text_iterator:
            if len(sentences) >= self.batch_size:
                break
            else:
                sentences.append(self.normalized_sentence(sentence))
                self.counter += 1
        return sentences

    def find_in_vocab(self, word):
        reverse_vocab = self.embedding_handler.word_to_index
        if word in reverse_vocab:
            return reverse_vocab[word]
        return reverse_vocab[self.embedding_handler.end_of_sentence_token]

    def normalized_sentence(self, sentence):
        sentence_arr = word_tokenize(sentence)

        sentence_indexes = [self.find_in_vocab(x.lower()) for x in sentence_arr]
        sentence_indexes.insert(0, self.embedding_handler.start_token_index)
        sentence_indexes.append(self.embedding_handler.end_token_index)
        sentence_indexes = sentence_indexes[:self.sentence_len+1]
        for idx in range(len(sentence_indexes), self.sentence_len + 1):
            sentence_indexes.append(self.embedding_handler.pad_token_index)

        return sentence_indexes
