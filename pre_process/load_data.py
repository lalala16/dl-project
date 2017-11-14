import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import sys
import os
import copy
import csv
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from utility.CsvUtility import CsvUtility

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def load_data(word_num_max=5000, sequence_max=100):
    x = CsvUtility.read_array_from_csv(origin_path + '/data/', 'x_train.csv').flatten()
    # print x[0]
    data = []
    for x_item in x:
        doc_data = []
        x_list = x_item.split('], [')
        x_list = [i.replace('[', '').replace(']', '') for i in x_list]
        # print x_list
        for seb_i, sen in enumerate(x_list):
            sen_list = [int(i) for i in sen.split(', ')]
            feature = [0.0] * word_num_max
            for i, item in enumerate(sen_list):
                if item < word_num_max:
                    feature[item] += 1
            doc_data.append(feature)
        if len(doc_data) < sequence_max:
            for _ in range(sequence_max - len(doc_data)):
                doc_data.append([0.0] * word_num_max)
        else:
            doc_data = doc_data[: sequence_max]
        # print doc_data
        data.append(doc_data)
    print 'shape of x: ', np.array(data).shape
    y = np.array(pd.read_csv(origin_path + '/data/y_train.csv', index_col=None, header=None))
    print 'shape of y: ', y.shape
    # print y
    train_data = Data.TensorDataset(data_tensor=torch.from_numpy(np.array(data)),
                                    target_tensor=torch.from_numpy(y))
    print 'Done'

    return train_data

class DatasetProcessingValidation(Dataset):
    def __init__(self, origin_path, x_train_path,word_num_max, sequence_max):
        f = open(os.path.join(origin_path, x_train_path), 'r')
        self.x = [line for line in f]
        f.close()
        print 'reading... shape of x: ', len(self.x)
        self.word_num_max = word_num_max
        self.sequence_max = sequence_max

    def __getitem__(self, index):
        x_item = self.x[index]
        doc_data = []
        x_list = x_item.split('], [')
        x_list = [i.replace('[', '').replace(']', '') for i in x_list]
        # print x_list
        for seb_i, sen in enumerate(x_list):
            sen_list = [int(i) for i in sen.split(', ')]
            feature = [0.0] * self.word_num_max
            for i, item in enumerate(sen_list):
                if item < self.word_num_max:
                    feature[item] += 1
            doc_data.append(feature)
        if len(doc_data) < self.sequence_max:
            for _ in range(self.sequence_max - len(doc_data)):
                doc_data.append([0.0] * self.word_num_max)
        else:
            doc_data = doc_data[: self.sequence_max]
        instance = torch.LongTensor(np.array(doc_data, dtype=np.int64))
        return instance

    def __len__(self):
        return len(self.x)

class DatasetProcessing(Dataset):
    def __init__(self, origin_path, x_train_path, y_train_path, word_num_max, sequence_max):
        f = open(os.path.join(origin_path, x_train_path), 'r')
        self.x = [line for line in f]
        f.close()
        print 'reading... shape of x: ', len(self.x)
        self.y = np.array(pd.read_csv(os.path.join(origin_path, y_train_path), index_col=None, header=None))
        print 'reading... shape of y: ', self.y.shape
        self.word_num_max = word_num_max
        self.sequence_max = sequence_max

    def __getitem__(self, index):
        x_item = self.x[index]
        doc_data = []
        x_list = x_item.split('], [')
        x_list = [i.replace('[', '').replace(']', '') for i in x_list]
        # print x_list
        for seb_i, sen in enumerate(x_list):
            sen_list = [int(i) for i in sen.split(', ')]
            feature = [0.0] * self.word_num_max
            for i, item in enumerate(sen_list):
                if item < self.word_num_max:
                    feature[item] += 1
            doc_data.append(feature)
        if len(doc_data) < self.sequence_max:
            for _ in range(self.sequence_max - len(doc_data)):
                doc_data.append([0.0] * self.word_num_max)
        else:
            doc_data = doc_data[: self.sequence_max]
        instance = torch.LongTensor(np.array(doc_data, dtype=np.int64))
        label = torch.LongTensor([self.y[index][0]])
        return instance, label

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    # tensor_d = load_data()
    f = open('../data/x_train.csv', 'r')
    cf = copy.deepcopy(f)
    f.close()
    for item in cf:
        print item

pass
