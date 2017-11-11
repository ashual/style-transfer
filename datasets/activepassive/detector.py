from nltk import sent_tokenize
import os
from datasets.activepassive.ispassive import Tagger

max_sen_len = 15
min_sen_len = 3


def is_valid_sentence(sent, min_words, max_words):
    sent_len = len(sent.strip(' \t\n\r').split())
    return min_words <= sent_len <= max_words


def analyze_file(source_filename):
    i = j = k = ln = 0
    source_file = open(source_filename, 'r', encoding='utf-8')
    print('analyzing %s \n' % source_filename)
    for line in source_file:
        ln += 1
        for sentence in sent_tokenize(line):
            i += 1
            if i % 100 == 0:
                print("%d lines read | %d valid sentences | %d active %d passive" % (ln, i, j, k))
            if is_valid_sentence(sentence, min_sen_len, max_sen_len):
                is_passive = t.is_passive(sentence)
                if not is_passive:
                    j += 1
                    active_file.write(sentence + os.linesep)
                else:
                    k += 1
                    passive_file.write(sentence + os.linesep)


def analyze_oanc():
    root = 'OANC-GrAF/data'
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('txt'):
                source_filename = os.path.join(subdir, file)
                analyze_file(source_filename)


active_file = open('active.text', 'a')
passive_file = open('passive.text', 'a')
t = Tagger()
# analyze_oanc()
analyze_file('plain_full_length.text')