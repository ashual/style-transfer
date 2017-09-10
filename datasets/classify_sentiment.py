# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn

import time
import json
import pickle

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

TRAIN_SIZE = 300000
TEST_SIZE = 3000


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


def classify(data):
    with open('classifier.obj', 'rb') as f:
        classifier = pickle.load(f)
    with open('vectorizer.obj', 'rb') as f:
        vectorizer = pickle.load(f)
    vectors = vectorizer.transform(data)
    return classifier.predict(vectors), classifier.decision_function(vectors)


def filter_sentences(is_positive, content):
    prediction, confidence = classify(content)
    filtered_content = []
    if is_positive:
        pred_should_be = 'pos'
    else:
        pred_should_be = 'neg'
    for i, sentence in enumerate(content):
        if prediction[i] == pred_should_be and abs(confidence[i]) >= .8 and len(word_tokenize(sentence)) >= 3:
            filtered_content.append(sentence)
    return filtered_content


def create_filtered_files():
    positive_content = [json.loads(s)['text'].lower() for s in get_positive_sentences()]
    negative_content = [json.loads(s)['text'].lower() for s in get_negative_sentences()]
    positive_content = filter_sentences(True, positive_content)
    negative_content = filter_sentences(False, negative_content)
    with open('pos.txt', 'w') as f:
        f.writelines("%s\n" % l for l in positive_content)
    with open('neg.txt', 'w') as f:
        f.writelines("%s\n" % l for l in negative_content)


def create_classifier():
    classes = ['pos', 'neg']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # for curr_class in classes:
    #     dirname = os.path.join(data_dir, curr_class)
    #     for fname in os.listdir(dirname):
    #         with open(os.path.join(dirname, fname), 'r') as f:
    #             content = f.read()
    #             if fname.startswith('cv9'):
    #                 test_data.append(content)
    #                 test_labels.append(curr_class)
    #             else:
    #                 train_data.append(content)
    #                 train_labels.append(curr_class)

    counter = 0
    print('loading data')
    negative_len = len(get_negative_sentences())
    positive_len = len(get_positive_sentences())
    negative_train_size = positive_train_size = int(min(negative_len, positive_len) * 0.9)
    negative_test_size = positive_test_size = int(min(negative_len, positive_len) * 0.1)
    # negative_train_size = int(negative_len * 0.9)
    # negative_test_size = negative_len - negative_train_size
    # positive_train_size = int(positive_len * 0.9)
    # positive_test_size = positive_len - positive_train_size
    print(negative_train_size, negative_test_size)
    print(positive_train_size, positive_test_size)

    for sen in get_negative_sentences():
        sen_json = json.loads(sen)

        if counter < negative_train_size:
            train_data.append(sen_json['text'])
            train_labels.append('neg')
        elif counter < negative_train_size + negative_test_size:
            test_data.append(sen_json['text'])
            test_labels.append('neg')
        else:
            break
        counter += 1

    counter = 0
    for sen in get_positive_sentences():
        sen_json = json.loads(sen)
        if counter < positive_train_size:
            train_data.append(sen_json['text'])
            train_labels.append('pos')
        elif counter < positive_train_size + positive_test_size:
            test_data.append(sen_json['text'])
            test_labels.append('pos')
        else:
            break
        counter += 1
    print('finish loading data {} {}'.format(len(train_data), len(test_data)))
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    with open('vectorizer.obj', 'wb') as file:
        pickle.dump(vectorizer, file)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    # classifier_rbf = svm.SVC()
    # t0 = time.time()
    # classifier_rbf.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_rbf = classifier_rbf.predict(test_vectors)
    # t2 = time.time()
    # time_rbf_train = t1-t0
    # time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    # classifier_linear = svm.SVC(kernel='linear')
    # t0 = time.time()
    # classifier_linear.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_linear = classifier_linear.predict(test_vectors)
    # t2 = time.time()
    # time_linear_train = t1-t0
    # time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1 - t0
    time_liblinear_predict = t2 - t1

    with open('classifier.obj', 'wb') as file:
        pickle.dump(classifier_liblinear, file)

    # Print results in a nice table
    # print("Results for SVC(kernel=rbf)")
    # print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    # print(classification_report(test_labels, prediction_rbf))
    # print("Results for SVC(kernel=linear)")
    # print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    # print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))

if __name__ == '__main__':
    pass
