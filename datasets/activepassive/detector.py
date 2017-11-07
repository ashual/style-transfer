import urllib.request
import urllib.parse
from nltk import sent_tokenize
# import re
import os

URL = 'https://datayze.com/supportcode/callback/passive.php'
# headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
# TODO: decide if to be sentences are passive or not for us

max_sen_len = 15
min_sen_len = 3


def get_voice(sen):
    values = {'txt': sen}
    data = urllib.parse.urlencode(values).encode("utf-8")
    req = urllib.request.Request(URL, data)
    response = urllib.request.urlopen(req)
    html = str(response.read())
    return '<li>' not in html

# def strip_sentence(sent):
#     return ' '.join(sent.strip(' \t\n\r').split())


def is_valid_sentence(sent, min_words, max_words):
    sent_len = len(sent.strip(' \t\n\r').split())
    # strip_sent = strip_sentence(sent)
    # return re.match(r'^[A-Z ]{1,9}: [A-Za-z,\?\.\ !\']*$', strip_sent) and min_words <= sent_len <= max_words
    return min_words <= sent_len <= max_words
#
#
# def normalize_sentence(sent):
#     return ' '.join(sent.split()[1:])
#
#
# def get_yoda_sentence(sent):
#     data = {'sentence': sent}
#     data_encoded = urllib.urlencode(data)
#
#     url = URL + '?' + data_encoded
#     req = urllib.request.urlopen(url, headers=headers)
#     max_tries = 3
#     while max_tries > 0:
#         try:
#             response = urllib.request.urlopen(req)
#             return response.read()
#         except urllib.error as e:
#             print(e.reason)
#             max_tries -= 1
#
#
# def remove_yoda_template(sent):
#     translation_w = re.sub(' Yeesssssss\.$', '', sent)
#     translation_w = re.sub(' Yes, hmmm\.$', '', translation_w)
#     translation_w = re.sub(' Herh herh herh\.$', '', translation_w)
#     translation_w = re.sub(' Hmmmmmm\.$', '', translation_w)
#     return re.sub(r', hmm', '', translation_w)
#


source_file = open('plain_full_length.text', 'r')
active_file = open('active.text', 'w')
passive_file = open('passive.text', 'w')

for line in source_file:
    for sentence in sent_tokenize(line):
        if is_valid_sentence(sentence, min_sen_len, max_sen_len):
            is_active_voice = get_voice(sentence)
            if is_active_voice:
                active_file.write(sentence + os.linesep)
            else:
                passive_file.write(sentence + os.linesep)


    # sentence = normalize_sentence(sentence)
    # yoda_sentence = get_yoda_sentence(sentence)
    # if not yoda_sentence:
    #     continue
    # yoda_sentence_strip = strip_sentence(yoda_sentence)
    # yoda_sentence_strip_lower = 'START ' + yoda_sentence_strip.lower()
    # yoda_sentence_without_lower = 'START ' + remove_yoda_template(yoda_sentence_strip).lower()
    # sentence_lower = 'START ' + sentence.lower()
    # if yoda_sentence_without_lower != sentence_lower:
    #     plain.write(sentence + '\n')
    #     yoda_english.write(yoda_sentence_strip_lower + '\n')
    #     yoda_english_without.write(yoda_sentence_without_lower + '\n')
    #     plain.flush()
    #     yoda_english.flush()
    #     yoda_english_without.flush()
