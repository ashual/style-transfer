import os
import re

import nltk.corpus.reader.bnc
from nltk import sent_tokenize

from datasets.activepassive.ispassive import Tagger

regexp = re.compile('^[A-Za-z][a-zA-Z\d ,?]*\.$')

max_sen_len = 15
min_sen_len = 4


def is_valid_sentence(sent):
    tokenize_sentence = nltk.sent_tokenize(sent.strip(' \t\n\r'))
    return is_valid_sentence_array(tokenize_sentence)


def is_valid_sentence_array(sent):
    if not min_sen_len <= len(sent) <= max_sen_len:
        return False
    for word in sent:
        if word.isupper():
            return False

    return regexp.search(' '.join(sent))


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
            if is_valid_sentence(sentence):
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


def analyze_bnc():
    i = j = k = 0
    # a = nltk.corpus.reader.bnc.BNCCorpusReader(root='Texts', fileids=r'[a-z]{3}/\w*\.xml')
    a = nltk.corpus.reader.bnc.BNCCorpusReader(root='2554/download/Texts', fileids=r'[A-K]/\w*/\w*\.xml')
    for sentence in a.sents():
        i += 1
        if i % 100 == 0:
            print("{} lines read | {} valid sentences | {} active {} passive".format(i, j + k, j, k))
        if is_valid_sentence_array(sentence):
            string_sentence = ' '.join(sentence)
            if t.is_passive(string_sentence):
                k += 1
                passive_file.write(string_sentence + os.linesep)
            else:
                j += 1
                active_file.write(string_sentence + os.linesep)


active_file = open('active.text', 'a')
passive_file = open('passive.text', 'a')
t = Tagger()
# analyze_oanc()
# analyze_file('plain_full_length.text')
analyze_bnc()
