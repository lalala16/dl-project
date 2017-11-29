import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import sys
import os
import re



sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.CsvUtility import CsvUtility

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def load_data(word_num_max=5000, sequence_max=100, word_sequence=100, valid_percent=0.2):
    # x = CsvUtility.read_array_from_csv(os.path.split(origin_path)[0] + '/fusai_data/', 'x_train.csv').flatten()
    f = open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'x_train.csv'), 'r')
    x = [line for line in f]
    f.close()
    # print x[0]
    data = np.zeros((len(x), sequence_max, word_sequence), dtype='int32')
    for i, sentences in enumerate(x):
        x_list = sentences.split('], [')
        # print len(x_list)
        x_list = [sen_i.replace('[', '').replace(']', '') for sen_i in x_list]
        for j, sent in enumerate(x_list):
            if j < sequence_max:
                sen_list = [int(re.sub("[^0-9]", "", sen_i).strip()) for sen_i in sent.split(', ')
                            if len(re.sub("[^0-9]", "", sen_i).strip()) > 0]
                k = 0
                for _, word in enumerate(sen_list):
                    if k < word_sequence and word <= word_num_max:
                        # print i,' ', j, ' ', k
                        data[i, j, k] = word
                        k = k + 1
    labels = np.array(pd.read_csv(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'y_train.csv'), index_col=None, header=None)).flatten()
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(valid_percent * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of positive and negative reviews in traing and validation set')
    print y_train.sum(axis=0)
    print y_val.sum(axis=0)

    f = open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'x_validation.csv'), 'r')
    x = [line for line in f]
    f.close()
    # print x[0]
    data = np.zeros((len(x), sequence_max, word_sequence), dtype='int32')
    for i, sentences in enumerate(x):
        x_list = sentences.split('], [')
        # print len(x_list)
        x_list = [sen_i.replace('[', '').replace(']', '') for sen_i in x_list]
        for j, sent in enumerate(x_list):
            if j < sequence_max:
                sen_list = [int(re.sub("[^0-9]", "", sen_i).strip()) for sen_i in sent.split(', ')
                            if len(re.sub("[^0-9]", "", sen_i).strip()) > 0]
                k = 0
                for _, word in enumerate(sen_list):
                    if k < word_sequence and word <= word_num_max:
                        data[i, j, k] = word
                        k = k + 1
    print('Shape of data tensor:', data.shape)
    data_val = np.array(pd.read_csv(os.path.split(origin_path)[0] + '/fusai_data/x_validation_id.csv', index_col=None, header=None))
    print 'shape of y: ', data_val.shape
    print 'Done'

    return x_train, y_train, x_val, y_val, data, data_val
if __name__ == '__main__':
    MAX_SENT_LENGTH = 100
    MAX_SENTS = 50
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    LSTM_HIDDEN_SIZE = 100

    EPOCH = 10
    BATCH_SIZE = 50

    x_train, y_train, x_val, y_val, data, data_val = load_data(
        word_num_max=MAX_NB_WORDS,
        sequence_max=MAX_SENTS,
        word_sequence=MAX_SENT_LENGTH,
        valid_percent=VALIDATION_SPLIT
    )