import numpy as np
import pandas as pd
import sys
import os
import re

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.CsvUtility import CsvUtility

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def load_data(word_num_max=12725, sequence_max=2048, train_perc=0.7):

    f = open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'x_train.csv'), 'r')
    x = [line for line in f]
    f.close()
    # print x[0]
    data = []
    for x_item in x:
        doc_data = [0] * sequence_max
        data_index = 0
        # print 'x: ', x_item
        x_list = x_item.split('], [')
        # print len(x_list)
        x_list = [i.replace('[', '').replace(']', '') for i in x_list]
        # print 'x list: ', x_list
        # print x_list
        for seb_i, sen in enumerate(x_list):
            if data_index >= sequence_max:
                break
            sen_list = [int(re.sub("[^0-9]", "", i).strip()) for i in sen.split(', ')
                        if len(re.sub("[^0-9]", "", i).strip()) > 0]
            # print 'sentence:', sen_list
            for i, item in enumerate(sen_list):
                if data_index >= sequence_max:
                    break
                if item < word_num_max:
                    doc_data[data_index] = item
                    data_index += 1
        # print doc_data
        data.append(doc_data)
    print 'shape of x: ', np.array(data).shape
    x = np.array(data)
    y = np.array(pd.read_csv(os.path.split(origin_path)[0] + '/fusai_data/y_train.csv', index_col=None, header=None))
    print 'shape of y: ', y.shape
    # print y
    print 'Done'
    train_size = int(x.shape[0] * train_perc)
    # shuffle the train set
    idx = np.random.permutation(x.shape[0])
    x_train = x[idx]
    y_train = y[idx]

    f = open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'x_validation.csv'), 'r')
    x = [line for line in f]
    f.close()
    # print x[0]
    data = []
    for x_item in x:
        doc_data = [0] * sequence_max
        data_index = 0
        # print 'x: ', x_item
        x_list = x_item.split('], [')
        # print len(x_list)
        x_list = [i.replace('[', '').replace(']', '') for i in x_list]
        # print 'x list: ', x_list
        # print x_list
        for seb_i, sen in enumerate(x_list):
            if data_index >= sequence_max:
                break
            sen_list = [int(re.sub("[^0-9]", "", i).strip()) for i in sen.split(', ')
                        if len(re.sub("[^0-9]", "", i).strip()) > 0]
            # print 'sentence:', sen_list
            for i, item in enumerate(sen_list):
                if data_index >= sequence_max:
                    break
                if item < word_num_max:
                    doc_data[data_index] = item
                    data_index += 1
        # print doc_data
        data.append(doc_data)
    print 'shape of x: ', np.array(data).shape
    x = np.array(data)
    y = np.array(pd.read_csv(os.path.split(origin_path)[0] + '/fusai_data/x_validation_id.csv', index_col=None, header=None))
    print 'shape of y: ', y.shape
    # print y
    print 'Done'
    # shuffle the train set
    CsvUtility.write2pickle(os.path.split(origin_path)[0] + '/fusai_data/pre_xtrain.pkl', x_train, 'w')
    CsvUtility.write2pickle(os.path.split(origin_path)[0] + '/fusai_data/pre_ytrain.pkl', y_train, 'w')
    CsvUtility.write2pickle(os.path.split(origin_path)[0] + '/fusai_data/pre_x.pkl', x, 'w')
    CsvUtility.write2pickle(os.path.split(origin_path)[0] + '/fusai_data/pre_y.pkl', y, 'w')
    return x_train, y_train, x, y

def reload_data():
    x_train = CsvUtility.read_pickle(os.path.split(origin_path)[0] + '/fusai_data/pre_xtrain.pkl', 'r')
    y_train = CsvUtility.read_pickle()
    x = CsvUtility.read_pickle()
    y = CsvUtility.read_pickle()
    return x_train, y_train, x, y
if __name__ == '__main__':
    print 'loading data...'
    x_train, y_train, x_test, x_id = load_data()
    print x_train[:5]
    print y_train[:5]
    print x_test[:3]
    print x_id[:3]