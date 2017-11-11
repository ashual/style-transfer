import urllib.request
import urllib.parse
from nltk import sent_tokenize
# import re
import os
import json

URL = 'https://datayze.com/supportcode/callback/passive.php'
# headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}
# TODO: decide if to be sentences are passive or not for us

max_sen_len = 15
min_sen_len = 3
max_tries = 3


def get_voice(sen):
    values = {'txt': sen}
    data = urllib.parse.urlencode(values).encode("utf-8")
    req = urllib.request.Request(URL, data)
    # tries = 0
    try:
        response = urllib.request.urlopen(req)
        # tries += 1
    except urllib.error.URLError as e:
        print(e.reason)
        return None
    html = str(response.read())
    if '<li>' not in html:
        return 'a'
    else:
        return 'p'


def is_valid_sentence(sent, min_words, max_words):
    sent_len = len(sent.strip(' \t\n\r').split())
    # strip_sent = strip_sentence(sent)
    # return re.match(r'^[A-Z ]{1,9}: [A-Za-z,\?\.\ !\']*$', strip_sent) and min_words <= sent_len <= max_words
    return min_words <= sent_len <= max_words

#
# def process_oanc():
#     for root, subdirs, files in os.walk():


status_file = open('status.text', 'r')
source_file = open('plain_full_length.text', 'r')
active_file = open('active.text', 'a')
passive_file = open('passive.text', 'a')
i = j = k = 0
# cur_line = int(status_file.readline())
# cur_line = 0
x = json.load(status_file)
if x:
    cur_line = int(x)
else:
    cur_line = 0
status_file.close()
status_file = open('status.text', 'w')
for i in range(cur_line):
    source_file.readline()
for line in source_file:
    cur_line += 1
    # status_file.write(str(cur_line))
    # json.dump(cur_line, status_file)
    for sentence in sent_tokenize(line):
        i += 1
        if i % 10 == 0:

            print("found ", k, " passive and ", j, " active valid sentences out of ", i,
                  " processed sentences")
        if is_valid_sentence(sentence, min_sen_len, max_sen_len):
            analyzed_voice = get_voice(sentence)
            if analyzed_voice == 'a':
                j += 1
                active_file.write(sentence + os.linesep)
            elif analyzed_voice == 'p':
                k += 1
                passive_file.write(sentence + os.linesep)

