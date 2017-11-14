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
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).double()
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).double()
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.softmax(self.hidden2label(lstm_out[-1]))
        return y


if __name__ == '__main__':

    # Hyper Parameters
    sequence_length = 100
    input_size = 5000
    embedding_dim = 100
    hidden_size = 128
    num_layers = 1
    num_classes = 2
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.01
    use_gpu = False

    # train_data = load_data.load_data(input_size, sequence_length)

    dtrain_set = prep.DatasetProcessing(origin_path+'/data/', 'x_train.csv',
                                        'y_train.csv', input_size, sequence_length)

    train_loader = DataLoader(dtrain_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )

    # Data Loader (Input Pipeline)
    # train_loader = torch.utils.data.DataLoader(dataset=train_data,
    #                                            batch_size=batch_size,
    #                                            shuffle=True, )


    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False)

    # create model
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_size,
                                 vocab_size=input_size, label_size=num_classes,
                                 batch_size=batch_size, use_gpu=use_gpu).double()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (instances, labels) in enumerate(train_loader):
            instances = Variable(instances.view(sequence_length, -1, input_size)).double()
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(instances)
            # print outputs
            # print labels.view(-1)
            loss = criterion(outputs, labels.view(-1))
            loss.backward(retain_graph=True)
            optimizer.step()

            if (i + 1) % 10 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1,  dtrain_set.__len__()// batch_size, loss.data[0]))

    '''
    # Test the Model
    dtest_set = prep.DatasetProcessing(origin_path + '/data/', 'x_test.csv',
                                       'y_test.csv', input_size, sequence_length)

    test_loader = DataLoader(dtest_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4
                             )
    correct = 0
    total = 0
    for i, (instances, labels) in enumerate(test_loader):
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
    dtest_set = prep.DatasetProcessingValidation(origin_path + '/data/', 'x_validation.csv',
                                       input_size, sequence_length)

    test_loader = DataLoader(dtest_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4
                             )
    pred_re = []
    for i, instances in enumerate(test_loader):

        instances = Variable(instances.view(sequence_length, -1, input_size)).double()
        outputs = model(instances)
        # print outputs.data
        _, predicted = torch.max(outputs.data, 1)
        pred_re.extend(predicted)
    # print len(pred_re)
    id_index = np.array(pd.read_csv(os.path.join(origin_path + '/data/', 'x_validation_id.csv'), header=None, index_col=None)).flatten()
    # print id_index
    with open(os.path.join(origin_path + '/data/', 'result.csv'), 'w') as f:
        for i, re in enumerate(pred_re):
            f.write(id_index[i]+','+str(re)+'\n')
