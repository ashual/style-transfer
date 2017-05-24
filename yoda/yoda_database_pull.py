import urllib
import urllib2
import re


URL = 'https://yoda.p.mashape.com/yoda'
KEY = 'RvyROurVRvmshRzDcAtXABdytvkDp1MtroWjsnXo8Dcs9Dq6SB'
headers = {'X-Mashape-Authorization': KEY, 'Accept': 'text/plain'}


def strip_sentence(sent):
    return ' '.join(sent.strip(' \t\n\r').split())


def is_valid_sentence(sent, min_words, max_words):
    sent_len = len(sent.strip(' \t\n\r').split()) - 1
    strip_sent = strip_sentence(sent)
    return re.match(r'^[A-Z ]{1,9}: [A-Za-z,\?\.\ !\']*$', strip_sent) and min_words <= sent_len <= max_words


def normalize_sentence(sent):
    return ' '.join(sent.split()[1:])


def get_yoda_sentence(sent):
    data = {'sentence': sent}
    data_encoded = urllib.urlencode(data)

    url = URL + '?' + data_encoded
    req = urllib2.Request(url, headers=headers)
    max_tries = 3
    while max_tries > 0:
        try:
            response = urllib2.urlopen(req)
            return response.read()
        except urllib2.URLError as e:
            print e.reason
            max_tries -= 1


def remove_yoda_template(sent):
    translation_w = re.sub(' Yeesssssss\.$', '', sent)
    translation_w = re.sub(' Yes, hmmm\.$', '', translation_w)
    translation_w = re.sub(' Herh herh herh\.$', '', translation_w)
    translation_w = re.sub(' Hmmmmmm\.$', '', translation_w)
    return re.sub(r', hmm', '', translation_w)

script = open('original.text', 'r')
yoda_english = open('yoda_english.text', 'w')
yoda_english_without = open('yoda_english_without_stupid_words.text', 'w')
for sentence in script:
    if not is_valid_sentence(sentence, 3, 10):
        continue
    sentence = normalize_sentence(sentence)
    yoda_sentence = get_yoda_sentence(sentence)
    if not yoda_sentence:
        continue
    yoda_sentence_strip = strip_sentence(yoda_sentence)
    yoda_sentence_without = remove_yoda_template(yoda_sentence_strip)

    if yoda_sentence_without.upper() != sentence.upper():
        yoda_english.write(sentence + '\n')
        yoda_english_without.write(sentence + '\n')
        yoda_english.write(yoda_sentence_strip + '\n')
        yoda_english_without.write(yoda_sentence_without + '\n')
        yoda_english.flush()
        yoda_english_without.flush()
