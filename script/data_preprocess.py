#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding:utf-8
import numpy as np
import pandas as pd
import jieba
import os
import sys
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.NlpUtility import serperate_text
from utility.WordDictUtility import Vocabulary
from utility.CsvUtility import CsvUtility


# get a small train data with 100 lines


def get_small_dataset(original_path='../data/', file_name='train.tsv'):
    with open(os.path.join(original_path, file_name), 'r') as rf:
        lines = rf.readlines()
        small_data = []
        for i in range(100):
            small_data.append(lines[i])

    with open(os.path.join(original_path, 'small_' + file_name), 'w') as wf:
        wf.writelines(small_data)



def load_corpus(all_path=['../data/small_train.tsv', '../data/evaluation_public.tsv'], train_perc=0.7):
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
            contend = serperate_text(data_line[1] + '。' + data_line[2])
            header_text_list.append(contend)
            if data_line[3] == 'POSITIVE':
                y.append(1)
            else:
                y.append(0)
            if count % 10000 == 0:
                print 'line ', count
    print 'data shape: ', len(header_text_list), len(y)

    voc_words = Vocabulary('../data/')
    voc_words.get_vocabulary(header_text_list, remove_stopwords=[], topnum=-1, save=True)

    x = []
    for i, line in enumerate(header_text_list):
        x.append(voc_words.get_feature_list(line))
    x = np.array(x)
    y = np.array(y)
    CsvUtility.write_array_csv(x, '../data/', 'x_train.csv', False)
    CsvUtility.write_array_csv(y, '../data/', 'y_train.csv', True)
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
    with open(all_path[1], 'r') as rf:
        rls = rf.readlines()
        for count, rline in enumerate(rls):
            data_line = rline.split('\t')
            # assert len(data_line) == 4
            if len(data_line) != 3:
                print 'innormal line:', count
            contend = serperate_text(data_line[1] + '。' + data_line[2])
            header_text_list.append(contend)

            if count % 10000 == 0:
                print 'line ', count
    print 'data shape: ', len(header_text_list), len(header_text_list[0])
    x = []
    for i, line in enumerate(header_text_list):
        x.append(voc_words.get_feature_list(line))
    x = np.array(x)
    CsvUtility.write_array_csv(x, '../data/', 'x_validation.csv', False)


if __name__ == '__main__':
    # load_corpus()
    get_small_dataset(file_name='evaluation_public.tsv')




