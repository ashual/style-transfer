

class EmbeddingHandler:
    def __init__(self):
        self.start_of_sentence_token = 'START'
        self.end_of_sentence_token = 'END'
        self.unknown_token = 'UNK'
        self.pad_token = 'PAD'

        self.word_to_index = None
        self.index_to_word = None
        self.embedding_np = None

    def vocabulary_to_internals(self, vocabulary):
        self.index_to_word = {i: w for i, w in enumerate(vocabulary)}
        self.word_to_index = {self.index_to_word[i]: i for i in self.index_to_word}

    def get_word_to_index(self, sentences):
        return [
            [
                self.word_to_index[w] if w in self.word_to_index else self.word_to_index[self.unknown_token] for w in s
            ]
            for s in sentences
        ]

    def get_index_to_word(self, sentences_with_indices):
        return [[self.index_to_word[i] for i in s] for s in sentences_with_indices]

    def get_vocabulary(self):
        return list(self.word_to_index.keys())

    def get_vocabulary_length(self):
        return len(self.word_to_index)

    def get_embedding_size(self):
        return self.embedding_np.shape[1]

    def get_embedding_array(self):
        return self.embedding_np
