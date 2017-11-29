from nltk import word_tokenize

# result = open('our_result_edit', 'w')
#
# with open('our_results.log') as f:
#     for line in f:
#         tokens = word_tokenize(line)
#         if tokens[-1] == 'END':
#             tokens = tokens[:-1]
#         last_word = 0
#         for i, word in enumerate(tokens):
#             if word not in ['.', '?', '!']:
#                 last_word = i
#         last_word = min(last_word + 2, len(tokens))
#         tokens = tokens[:last_word]
#         result.write(' '.join(tokens) + '\n')


regina = open('sentiment.test.0.tsf', 'r')
our = open('our_result_edit.log', 'r')
original = open('./datasets/yelp/regina-data/sentiment.test.0', 'r')

with open('combined_results.log', 'w') as f:
    for reg_s in regina:
        our_s = our.readline()
        ori_s = original.readline()
        f.write('ori: {}our: {}reg: {}\n'.format(ori_s, our_s, reg_s))