import csv


def read_csv(file_name):
    f = open(file_name, 'rb')
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
                data[key] = {first_row[i]: row[i] for i in xrange(len(first_row))}
        return data


class Bibles:
    def __init__(self, first, second):
        bibles = csv_to_dict('bible-corpus/bible_version_key.csv', 'table')
        if first not in bibles or second not in bibles:
            print('Please use only legal bible names:')
            Bibles.print_all_options()
            exit(1)
        self.first_bible = read_csv("bible-corpus/{}.csv".format(first))
        self.second_bible = read_csv("bible-corpus/{}.csv".format(second))
        self.first_bible.next()
        self.second_bible.next()

    def __iter__(self):
        return self

    def __next__(self):
        return self.first_bible.next()[4], self.second_bible.next()[4]

    @staticmethod
    def print_all_options():
        dict = csv_to_dict('bible-corpus/bible_version_key.csv')
        for row in dict.values():
            print(row['table'], row['version'], row['info_url'])

if 'name' == '__main__':
    for idx, (first, second) in enumerate(Bibles('t_asv', 't_ylt')):
        print(first, second)
        if idx > 10:
            break

    Bibles.print_all_options()