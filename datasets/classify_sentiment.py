import json
import pickle
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

def extract_non_indifferent_sentences(max_sentences=10, max_words=15):
    '''
    Get sentences from the review dataset with 1/2/4/5 stars and
    maximum sentences in a text, and only maximum of max_words in a sentence
    :param max_sentences: If the text exceed this number of sentences do not use this text
    :param max_words: Return only sentences with maximum of 15 words
    :return: create two files positive_reviews and negative_reviews
    '''
    positive_reviews = open('yelp/positive_reviews.json', 'wb')
    negative_reviews = open('yelp/negative_reviews.json', 'wb')
    with open('yelp/yelp_academic_dataset_review.json') as yelp:
        content = yelp.readlines()
    for idx, review in enumerate(content):
        try:
            j_review = json.loads(review)
            stars_review = j_review['stars']
            text_review = j_review['text']
        except ValueError:
            continue
        except KeyError:
            continue
        if stars_review == 3:
            continue
        text_review_sentences = text_review.split('.')
        text_review_sentences = filter(None, text_review_sentences)
        if len(text_review_sentences) > max_sentences:
            continue
        if stars_review > 3:
            write_to = positive_reviews
        else:
            write_to = negative_reviews
        for sent in text_review_sentences:
            if len(sent.split()) > max_words:
                continue
            try:
                formatted_sent = '{}.'.format(' '.join(sent.split()))
            except UnicodeEncodeError:
                continue
            compact_json = {'stars': stars_review, 'text': formatted_sent}
            json.dump(compact_json, write_to)
            write_to.write('\n')
    positive_reviews.close()
    negative_reviews.close()


def get_positive_sentences():
    '''
    After calling this function, one should iterate the content and for each line
    call json.loads(sentence)
    the object contains stars and text
    :return: line iterator
    '''
    with open('yelp/positive_reviews.json') as yelp:
        content = yelp.readlines()
    return content

def get_negative_sentences():
    '''
    After calling this function, one should iterate the content and for each line
    call json.loads(sentence)
    the object contains stars and text
    :return: line iterator
    '''
    with open('yelp/negative_reviews.json') as yelp:
        content = yelp.readlines()
    return content

def classify_sentence(sentence, cl):
    blob = TextBlob(sentence, classifier=cl)
    for s in blob.sentences:
        print(s)
        print(s.classify())


TRAIN_SIZE = 500000
TEST_SIZE = 1000

def get_train_and_test_datasets():
    train = []
    test = []
    counter = 0
    for sen in get_negative_sentences():
        a = json.loads(sen)
        sample = (a['text'], 'neg')

        if counter < TRAIN_SIZE:
            train.append(sample)
        elif counter < TRAIN_SIZE + TEST_SIZE:
            test.append(sample)
        else:
            break
        counter += 1

    counter = 0
    for sen in get_positive_sentences():
        a = json.loads(sen)
        sample = (a['text'], 'pos')
        if counter < TRAIN_SIZE:
            train.append(sample)
        elif counter < TRAIN_SIZE + TEST_SIZE:
            test.append(sample)
        else:
            break
        counter += 1
    return train, test

def train_and_test():
    train, test = get_train_and_test_datasets()
    print('training')
    cl = NaiveBayesClassifier(train)
    print(cl.accuracy(test))
    with open('classifier.obj', 'wb') as file:
        pickle.dump(cl, file)

def format_json():
    train = []
    counter = 0
    with open('negative.json', 'w') as negative:
        for sen in get_negative_sentences():
            sen = json.loads(sen)
            obj = {'text': sen['text'], 'label': 'neg'}
            train.append(obj)
            if counter > 1000:
                break
        json.dump(train, negative, indent=True)

def test():
    _, test = get_train_and_test_datasets()
    with open('classifier.obj', 'rb') as handle:
        cl = pickle.load(handle)
    print('Computing accuracy')
    acc = cl.accuract(test)
    print('Accuracy: {}'.format(acc))

#format_json()
# with open('negative.json', 'r') as fp:
#     cl = NaiveBayesClassifier(fp, format='json')

#train_and_test()
test()