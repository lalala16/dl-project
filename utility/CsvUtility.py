# -*- coding:utf-8 -*-
import re
import nltk
import nltk.data
import pandas as pd
import os
import sys
import cPickle as CPickle
import pickle
import numpy as np
import csv
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
class CsvUtility(object):

    @staticmethod
    def text2words(raw_text, stop_words=[], stem_word=False):

        # Remove non-letters
        text = re.sub("[^a-zA-Z]", " ", raw_text)

        # Convert words to lower case and split them
        words = text.lower().split()

        # Optionally remove stop words (false by default)
        words = [w for w in words if not w in stop_words]

        # Optionally stem words
        if stem_word:
            words = [nltk.PorterStemmer().stem(w) for w in words]
        return words

    @staticmethod
    def text2sentence(raw_text, token, stop_words=[], stem_word=False):

        # use the NLTK tokenizer to split the paragraph into sentences
        # for saving time, delete the token here
        # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        raw_sentences = token.tokenize(raw_text.decode('utf8').strip())

        # loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(CsvUtility.text2words(raw_text=raw_sentence,
                                                       stop_words=stop_words,
                                                       stem_word=stem_word))
        return sentences

    @staticmethod
    def write2pickle(file_path, object, type):
        with open(file_path, type) as f:
            CPickle.dump(object, f)
            f.close()

    @staticmethod
    def read_pickle(file_path, type):
        with open(file_path, type) as f:
            obj = CPickle.load(f)
            f.close()
        return obj

    @staticmethod
    def write_dict2csv(raw_dict, csv_path, file_name):
        pd.DataFrame.from_dict(raw_dict, orient='index').to_csv(os.path.join(csv_path, file_name), header=False)

    @staticmethod
    def write_key_value_times(raw_dict, csv_path, file_name):
        # try:
        #     os.makedirs(os.path.join(csv_path, file_name))
        # except Exception:
        #     pass
        with open(csv_path+'/'+file_name, 'w') as f:
            write_str = ""
            for (key, value) in raw_dict.items():
                for i in range(value):
                    write_str += key+","
            f.write(write_str)
        return len(raw_dict)

    @staticmethod
    def read_array_from_csv(csv_path, file_name, type='float'):
        re_array = np.array(pd.read_csv(os.path.join(csv_path, file_name), header=None, index_col=None))
        if re_array.shape[0] == 1:
            return re_array.flatten()
        return re_array

    @staticmethod
    def write_array_csv_test(raw_array, csv_path, file_name):
        np_raw = np.array(raw_array)
        if np_raw.ndim == 1:
            np_raw = np_raw.reshape(len(raw_array), -1)
        pd.DataFrame(np_raw).to_csv(os.path.join(csv_path, file_name), index=None, header=None)


if __name__ == '__main__':

    tests = ["dd", "df3ef", "dfeww"]
    print tests
    CsvUtility.write2pickle('../data-repository/123.pkl', tests, 'w')
    # CsvUtility.write_list2csv(tests, '../data-repository', 'test_csvutility.csv')
    # a = np.random.permutation(10)
    # print b[a]
    # print a
    # CsvUtility.write_array2csv(np.array(b), '../data-repository', 'test_csvutility.csv')
    # re_a = CsvUtility.read_array_from_csv('../data-repository', 'test_csvutility.csv')
    # print re_a
    # print re_a.shape
    # print b[re_a]
    word_dict = CsvUtility.read_pickle('../data-repository/event_dict.pickle', 'r')
    print word_dict