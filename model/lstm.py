import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

import pre_process.load_data as prep

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Linear(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.softmax = nn.Softmax()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()).double()
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()).double()
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).double()
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).double()
        return (h0, c0)

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        # print embeds[0]
        lstm_out, _ = self.lstm(embeds, self.hidden)
        # print self.hidden[0].size(), self.hidden[1].size()
        # print lstm_out[-1]
        return self.softmax(self.hidden2label(lstm_out[-1]))


if __name__ == '__main__':

    # Hyper Parameters
    sequence_length = 100
    input_size = 10000
    embedding_dim = 256
    hidden_size = 64
    num_layers = 1
    num_classes = 2
    batch_size = 20
    num_epochs = 30
    learning_rate = 0.01
    use_gpu = False

    # train_data = load_data.load_data(input_size, sequence_length)

    dtrain_set = prep.DatasetProcessing(os.path.split(origin_path)[0] + '/data/', 'x_train.csv',
                                        'y_train.csv', input_size, sequence_length)

    train_loader = DataLoader(dtrain_set,
                              batch_size=batch_size,
                              shuffle=True,
                              )

    # create model
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_size,
                                 vocab_size=input_size, label_size=num_classes,
                                 batch_size=batch_size, use_gpu=use_gpu).double()
    if use_gpu:
        '''
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        '''
        if torch.cuda.is_available():
            model.cuda()
    # Loss and Optimizer
    if use_gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # print '**********************************'
    # print [i.size() for i in model.parameters()]
    # print '**********************************'
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (instances, labels) in enumerate(train_loader):
            # print '----------------batch------'
            if use_gpu:
                instances = Variable(instances.view(sequence_length, -1, input_size).cuda()).double()
                # print torch.squeeze(labels).shape
                # print labels.shape
                labels = Variable(torch.squeeze(labels).cuda())
            else:
                instances = Variable(instances.view(sequence_length, -1, input_size)).double()
                labels = Variable(torch.squeeze(labels))
            #print instances_v
            #print labels_v
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(instances)
            # print outputs, labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1000 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1,  dtrain_set.__len__()// batch_size, loss.data[0]))

    '''
    # Test the Model
    dtest_set = prep.DatasetProcessing(os.path.split(origin_path)[0] + '/data/', 'x_test.csv',
                                       'y_test.csv', input_size, sequence_length)

    test_loader = DataLoader(dtest_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    correct = 0
    total = 0
    for i, (instances, labels) in enumerate(test_loader):
        if use_gpu:
            instances = Variable(instances.view(sequence_length, -1, input_size).cuda()).double()
        else:
            instances = Variable(instances.view(sequence_length, -1, input_size)).double()
        outputs = model(instances)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == (labels.view(predicted.shape))).sum()
        print('Test Accuracy of the model: %d %%' % (100 * correct / total))

    print('Test Accuracy of the model: %d %%' % (100 * correct / total))
    '''
    # Save the Model
    torch.save(model.state_dict(), 'rnn.pkl')

    # Predict the Result
    dtest_set = prep.DatasetProcessingValidation(os.path.split(origin_path)[0] + '/data/', 'x_validation.csv',
                                       input_size, sequence_length)

    test_loader = DataLoader(dtest_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )
    pred_re = []
    id_index = np.array(pd.read_csv(os.path.join(os.path.split(origin_path)[0] + '/data/', 'x_validation_id.csv'), header=None, index_col=None)).flatten()
    index_re = []
    for i, (instances, i_index) in enumerate(test_loader):

        if use_gpu:
            instances = Variable(instances.view(sequence_length, -1, input_size).cuda()).double()
        else:
            instances = Variable(instances.view(sequence_length, -1, input_size)).double()
        outputs = model(instances)
        # print outputs.data
        _, predicted = torch.max(outputs.data, 1)
        pred_re.extend(predicted)
        index_re.extend(i_index)
        if (i + 1) % 100 == 0:
            print 'predict :', i
    # print index_re
    id_index = id_index[index_re]

    # print id_index
    with open(os.path.join(os.path.split(origin_path)[0] + '/data/', 'result.csv'), 'w') as f:
        for i, re in enumerate(pred_re):
            f.write(id_index[i]+','+str(re)+'\n')
