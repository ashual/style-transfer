from nltk import word_tokenize
from random import shuffle

from datasets.batch import Batch


class BatchIterator:
    def __init__(self, content, embedding_handler, sentence_len, batch_size, shuffle_sentences=True):
        self.content = content
        self.embedding_handler = embedding_handler
        self.sentence_len = sentence_len
        self.batch_size = batch_size
        self.shuffle_sentences = shuffle_sentences

    def __iter__(self):
        if self.shuffle_sentences:
            shuffle(self.content)
        res = Batch()
        for sentence in self.content:
            if res.get_len() >= self.batch_size:
                yield res
                res = Batch()
            # append the current sentence
            sentences, lengths = self.normalized_sentence(sentence)
            res.add(sentences, lengths)
        if res.get_len() > 0:
            yield res

    def normalized_sentence(self, sentence):
        # get the words in lower case + and end tokens
        sentence_arr = [x.lower() for x in word_tokenize(sentence)]
        sentence_arr.append(self.embedding_handler.end_of_sentence_token)
        # cut to the allowed size
        sentence_arr = sentence_arr[:self.sentence_len]
        sentence_arr = self.embedding_handler.get_word_to_index([sentence_arr])[0]
        sentence_length = len(sentence_arr)
        # add padding if needed
        padding_length = (self.sentence_len - sentence_length)
        # the padding index would be like extending the vocabulary by one, in the embedding matrix of the embedding
        # translator we add a constant 0 row for this index
        pad_index = self.embedding_handler.get_vocabulary_length()
        # right
        sentence_arr = sentence_arr + [pad_index] * padding_length
        # return as indices sentence
        return sentence_arr, sentence_length
