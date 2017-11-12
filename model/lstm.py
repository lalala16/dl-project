import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from pre_process import load_data


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

    train_data = load_data.load_data(input_size, sequence_length)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True, )

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
            loss.backward(retain_variables=True)
            optimizer.step()

            if (i + 1) % 2 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.data[0]))

    # Test the Model
    '''
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
    '''
    # Save the Model
    torch.save(model.state_dict(), 'rnn.pkl')
