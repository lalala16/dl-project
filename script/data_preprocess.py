#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding:utf-8
import numpy as np
import pandas as pd
import jieba
import os
import sys
import itertools
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.NlpUtility import serperate_text
from utility.WordDictUtility import Vocabulary
from utility.CsvUtility import CsvUtility


# get a small train data with 100 lines
origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def get_small_dataset(original_path='../data/', file_name='train.tsv'):
    original_path = os.path.split(origin_path)[0] + '/data/'
    with open(os.path.join(original_path, file_name), 'r') as rf:
        lines = rf.readlines()
        small_data = []
        for i in range(1000):
            small_data.append(lines[i])

    with open(os.path.join(original_path, 'small_' + file_name), 'w') as wf:
        wf.writelines(small_data)



def load_corpus(origin_path, all_path=['../data/train.tsv', '../data/evaluation_public.tsv'], have_sentence=False, train_perc=0.7):
    #data = np.array(pd.read_csv(all_path, sep='\t', header=None, index_col=None))
    # print data[:5]

    header_text_list = []
    y = []
    with open(all_path[0], 'r') as rf:
        rls = rf.readlines()
        for count, rline in enumerate(rls):
            data_line = rline.split('\t')
            # assert len(data_line) == 4
            if len(data_line) != 4:
                print 'innormal line:', count
            contend = serperate_text(data_line[1] + '。' + data_line[2], text2sentence=have_sentence)
            # print contend
            header_text_list.append(contend)
            # print data_line[3].strip()=="POSITIVE"
            if data_line[3].strip() == 'POSITIVE':
                y.append(1)
            else:
                y.append(0)
            if count % 10000 == 0:
                print 'line ', count
    print 'data shape: ', len(header_text_list), len(y)

    voc_words = Vocabulary(origin_path + '/fusai_data/')
    voc_words.get_vocabulary(header_text_list, remove_stopwords=[], topnum=-1, save=True, have_sentence=have_sentence)

    x = []
    for i, line in enumerate(header_text_list):
        x.append(voc_words.get_feature_list(line, have_sentence=have_sentence))
    if have_sentence:
        CsvUtility.write_array_csv_test(x, origin_path + '/fusai_data/', 'x_train_sentence.csv')
        CsvUtility.write_array_csv_test(y, origin_path + '/fusai_data/', 'y_train_sentence.csv')
    else:
        CsvUtility.write_array_csv_test(x, origin_path + '/fusai_data/', 'x_train.csv')
        CsvUtility.write_array_csv_test(y, origin_path + '/fusai_data/', 'y_train.csv')
    '''
    train_size = int(x.shape[0] * train_perc)
    # shuffle the train set
    idx = np.random.permutation(x.shape[0])
    x_train = x[idx]
    y_train = y[idx]
    return x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:]
    '''
    # preprocess validation set

    header_text_list = []
    text_id_list = []
    with open(all_path[1], 'r') as rf:
        rls = rf.readlines()
        for count, rline in enumerate(rls):
            data_line = rline.split('\t')
            # assert len(data_line) == 4
            if len(data_line) != 3:
                print 'innormal line:', count
            contend = serperate_text(data_line[1] + '。' + data_line[2], text2sentence=have_sentence)
            header_text_list.append(contend)
            text_id_list.append(data_line[0])
            if count % 10000 == 0:
                print 'line ', count
    print 'data shape: ', len(header_text_list), len(header_text_list[0]), len(text_id_list)
    x = []

    for i, line in enumerate(header_text_list):
        x.append(voc_words.get_feature_list(line, have_sentence=have_sentence))
    if have_sentence:
        CsvUtility.write_array_csv_test(x, origin_path + '/fusai_data/', 'x_validation_sentence.csv')
    else:
        CsvUtility.write_array_csv_test(x, origin_path + '/fusai_data/', 'x_validation.csv')
    CsvUtility.write_array_csv_test(text_id_list, origin_path + '/fusai_data/', 'x_validation_id.csv')
    '''
    text_id_list = []
    with open(all_path[1], 'r') as rf:
        rls = rf.readlines()
        for count, rline in enumerate(rls):
            data_line = rline.split('\t')
            text_id_list.append(data_line[0])
            if count % 10000 == 0:
                print 'line ', count
    CsvUtility.write_array_csv_test(text_id_list, origin_path + '/data/', 'x_validation_id.csv')
    '''

if __name__ == '__main__':
    # print sys.path
    # origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    load_corpus(os.path.split(origin_path)[0], [os.path.split(origin_path)[0] + '/fusai_data/train.tsv', os.path.split(origin_path)[0] + '/fusai_data/evaluation_public.tsv'], have_sentence=True)
    # get_small_dataset(file_name='evaluation_public.tsv')




