import collections
import zipfile
import tensorflow as tf
import json


class WordIndexer:
    def __init__(self, filename, n=1, truncate_by_cutoff=True):
        print('word indexing: initializing')
        vocabulary = self.read_data(filename)
        print('word indexing: data size', len(vocabulary))
        print('word indexing: processing')
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset(vocabulary, n,
                                                                                             truncate_by_cutoff)
        del vocabulary
        print('word indexing: dictionaries ready')

    @staticmethod
    def read_data(filename):
        """Extract the first file enclosed in a zip file as a list of words."""
        if filename[-3:] == 'zip':
            with zipfile.ZipFile(filename) as f:
                data = tf.compat.as_str(f.read(f.namelist()[0])).split()
                print(data[:200])
        elif 'reviews' in filename:
            jdec = json.JSONDecoder()
            data = []
            with open(filename) as f:
                for line in f:
                    sentence = "%s %s %s" % ('START', jdec.decode(line)['text'][:-1], 'STOP')
                    data.extend(sentence.split())
                    # data += "%s %s %s " % ('START', sentence, 'STOP') #very very bad?
            # print(data[:1000])
            f.close()
        return data

    # Step 2: Build the dictionary and replace rare words with UNK token.
    @staticmethod
    def build_dataset(words, n, truncate_by_cutoff):
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]]
        dictionary = dict()
        if truncate_by_cutoff:
            count.extend(collections.Counter(words).most_common())
            for w, c in count:
                if c > n:
                    dictionary[w] = len(dictionary)
        else:
            count.extend(collections.Counter(words).most_common(n - 1))
            for word, _ in count:
                dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    def ind2word(self, ind):
        if ind in self.reverse_dictionary.keys():
            return self.reverse_dictionary[ind]
        else:
            return 'UNK'

    def word2ind(self, word):
        if word in self.dictionary.keys():
            return self.dictionary[word]
        else:
            return len(self.dictionary)

    def sample_indices(self):
        print('Most common words (+UNK)', self.count[:5])
        print('Sample data', self.data[:10], [self.reverse_dictionary[i] for i in self.data[:10]])
        print('UNK words will return', self.word2ind('asdhbvalksdvbaslkjdvb'))
