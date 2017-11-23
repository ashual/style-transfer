from nltk import word_tokenize

regular = open('regular.txt', 'w')
simple = open('simple.txt', 'w')
more_1 = 0
more_2 = 0
with open('parawiki_english05') as f:
    for sent in f:
        split = sent.split('\t')
        if len(word_tokenize(split[0])) > 15:
            more_1 += 1
        else:
            regular.write(split[0] + '\n')
        if len(word_tokenize(split[1])) > 15:
            more_2 += 1
        else:
            simple.write(split[1] + '\n')

print(more_1)
print(more_2)
