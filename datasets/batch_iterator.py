from nltk import word_tokenize
from random import shuffle

from datasets.batch import Batch


class BatchIterator:
    def __init__(self, dataset, embedding_handler, sentence_len, batch_size, clip_redundant_padding=True):
        self.dataset = dataset
        self.embedding_handler = embedding_handler
        self.sentence_len = sentence_len
        self.batch_size = batch_size
        self.text_iterator = None
        self.clip_redundant_padding = clip_redundant_padding

    def __iter__(self):
        content = self.dataset.get_content()
        shuffle(content)
        self.text_iterator = iter(content)
        return self

    def __next__(self):
        res = Batch()
        for sentence in self.text_iterator:
            if res.get_len() >= self.batch_size:
                break
            else:
                left_sentence, left_mask, right_sentence, right_mask = self.normalized_sentence(sentence)
                res.add(left_sentence, left_mask, right_sentence, right_mask)
        if res.get_len() == 0:
            raise StopIteration
        if self.clip_redundant_padding:
            res.clip_redundant_padding(res.get_removable_pads())
        return res

    def normalized_sentence(self, sentence):
        # get the words in lower case + and end tokens
        sentence_arr = [x.lower() for x in word_tokenize(sentence)]
        sentence_arr.append(self.embedding_handler.end_of_sentence_token)
        # cut to the allowed size
        sentence_arr = sentence_arr[:self.sentence_len]
        sentence_arr = self.embedding_handler.get_word_to_index([sentence_arr])[0]
        padding_arr = [1] * len(sentence_arr)
        # add padding if needed
        padding_length = (self.sentence_len - len(sentence_arr))
        # the padding index would be like extending the voabulary by one, in the embedding matrix of the embedding
        # translator we add a constatnt 0 row for this index
        pad_index = self.embedding_handler.get_vocabulary_length()
        # left
        left_sentence_arr = [pad_index] * padding_length + sentence_arr
        left_padding_arr = [0] * padding_length + padding_arr
        # right
        right_sentence_arr = sentence_arr + [pad_index] * padding_length
        right_padding_arr = padding_arr + [0] * padding_length
        # return as indices sentence
        return left_sentence_arr, left_padding_arr, right_sentence_arr, right_padding_arr
