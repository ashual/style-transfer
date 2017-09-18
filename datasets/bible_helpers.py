import csv
import itertools
from datasets.dataset import Dataset
from nltk import sent_tokenize


def read_csv(file_name):
    f = open(file_name, 'r')
    return csv.reader(f)


def csv_to_dict(file_name, default_key=None):
    data = dict()

    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        for idx, row in enumerate(reader):
            if idx == 0:
                first_row = row
            else:
                if default_key:
                    key = row[first_row.index(default_key)]
                else:
                    key = idx
                data[key] = {first_row[i]: row[i] for i in range(len(first_row))}
        return data


class Bibles(Dataset):
    def __init__(self, name, limit_sentences, dataset_cache_dir, dataset_name):
        Dataset.__init__(self, limit_sentences, dataset_cache_dir, dataset_name)
        bibles = csv_to_dict('datasets/bible-corpus/bible_version_key.csv', 'table')
        if name not in bibles:
            print('Please use only legal bible names:')
            Bibles.print_all_options()
            exit(1)
        self.bible = read_csv("datasets/bible-corpus/{}.csv".format(name))

    def get_content_actual(self):
        content = [sent_tokenize(sentence[4]) for sentence in self.bible]
        flatten = list(itertools.chain.from_iterable(content))
        return flatten[1:-1]

    @staticmethod
    def print_all_options():
        dict = csv_to_dict('datasets/bible-corpus/bible_version_key.csv')
        for row in dict.values():
            print(row['table'], row['version'], row['info_url'])

if 'name' == '__main__':
    for idx, (first, second) in enumerate(Bibles('t_asv', 't_ylt')):
        print(first, second)
        if idx > 10:
            break

    Bibles.print_all_options()