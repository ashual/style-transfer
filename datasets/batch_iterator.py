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

    def normalized_sentence(self, sentence):
        # get the words in lower case + start and end tokens
        sentence_arr = [self.embedding_handler.start_of_sentence_token] + \
                       [x.lower() for x in word_tokenize(sentence)] + \
                       [self.embedding_handler.end_of_sentence_token]
        # cut to the allowed size
        sentence_arr = sentence_arr[:self.sentence_len+1]
        # add padding if needed
        sentence_arr += [self.embedding_handler.pad_token] * (self.sentence_len + 1 - len(sentence_arr))
        # return as indices sentence
        return self.embedding_handler.get_word_to_index([sentence_arr])[0]
