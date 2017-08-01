from nltk import word_tokenize
from random import shuffle


class BatchIterator:
    def __init__(self, dataset, embedding_handler, sentence_len, batch_size):
        self.dataset = dataset
        self.embedding_handler = embedding_handler
        self.sentence_len = sentence_len
        self.batch_size = batch_size
        self.text_iterator = None

    def __iter__(self):
        content = self.dataset.get_content()
        shuffle(content)
        self.text_iterator = iter(content)
        return self

    def __next__(self):
        sentences = []
        for sentence in self.text_iterator:
            if len(sentences) >= self.batch_size:
                break
            else:
                sentences.append(self.normalized_sentence(sentence))
        if len(sentences) == 0:
            raise StopIteration
        return sentences

    def normalized_sentence(self, sentence):
        # get the words in lower case + start and end tokens
        sentence_arr = [x.lower() for x in word_tokenize(sentence)]
        sentence_arr.append(self.embedding_handler.end_of_sentence_token)
        # cut to the allowed size
        sentence_arr = sentence_arr[:self.sentence_len]
        # add padding if needed
        sentence_arr += [self.embedding_handler.pad_token] * (self.sentence_len - len(sentence_arr))
        # return as indices sentence
        return self.embedding_handler.get_word_to_index([sentence_arr])[0]
