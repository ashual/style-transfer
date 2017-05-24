import urllib
import urllib2
import re


URL = 'https://yoda.p.mashape.com/yoda'
KEY = 'RvyROurVRvmshRzDcAtXABdytvkDp1MtroWjsnXo8Dcs9Dq6SB'
headers = {'X-Mashape-Authorization': KEY, 'Accept': 'text/plain'}

script = open('original.text', 'r')
yoda_english = open('yoga_english.text', 'w')
yoda_english_without = open('yoga_english_without_stupid_words.text', 'w')
for sentence in script:
    if not re.match(r'^[A-Z]{1,7}', sentence):
        continue
    sentence = re.sub(r'^[A-Z ]*: ', "", sentence)[:-1]
    data = {'sentence': sentence}
    data_encoded = urllib.urlencode(data)

    url = URL + '?' + data_encoded
    req = urllib2.Request(url, headers=headers)
    success = False
    max_tries = 3
    print req.get_full_url()
    while not success and max_tries > 0:
        try:
            response = urllib2.urlopen(req)
            translation = response.read()
            translation_w = re.sub('  Yeesssssss\.$', '', translation)
            translation_w = re.sub('  Yes, hmmm\.$', '', translation_w)
            translation_w = re.sub('  Herh herh herh\.$', '', translation_w)
            translation_w = re.sub('  Hmmmmmm\.$', '', translation_w)
            translation_w = re.sub(r', hmm', '', translation_w)
            print translation_w, sentence, translation_w != sentence
            if translation_w != sentence:
                yoda_english.write(sentence + '\n')
                yoda_english_without.write(sentence + '\n')
                yoda_english.write(translation + '\n')
                yoda_english_without.write(translation_w + '\n')
                yoda_english.flush()
                yoda_english_without.flush()
            success = True
        except urllib2.URLError as e:
            print e.reason
            print e
            yoda_english_without.write('error#:' + translation_w + '\n')
            yoda_english.write('error#:' + translation_w + '\n')
            max_tries -= 1
