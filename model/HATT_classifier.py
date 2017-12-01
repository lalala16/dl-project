import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

import sys
import os
import pickle

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import save_model,load_model
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from gensim.models import Word2Vec

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from pre_process import load_data_HATT

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

MAX_SENT_LENGTH = 100
MAX_SENTS = 100
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

LSTM_HIDDEN_SIZE = 64

EPOCH = 5
BATCH_SIZE = 64


'''
GLOVE_DIR = "/ext/home/analyst/Testground/data/glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))
'''
with open(os.path.join(os.path.split(origin_path)[0], 'fusai_data/dictionary.pkl')) as f:
    word_index = pickle.load(f)
    index_word = pickle.load(f)

    embedding_model = Word2Vec.load(os.path.join(os.path.split(origin_path)[0], 'fusai_data/my.model'))
    embedding_matrix = np.random.random((MAX_NB_WORDS + 1, EMBEDDING_DIM))
    count = 0
    for word, i in word_index.items():
        # print word
        if i <= MAX_NB_WORDS and word in embedding_model:
            embedding_vector = embedding_model[word]
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count += 1
    print 'find embedding word:', count

x_train, y_train, data, data_val = load_data_HATT.load_data(
    word_num_max=MAX_NB_WORDS,
    sequence_max=MAX_SENTS,
    word_sequence=MAX_SENT_LENGTH,
    valid_percent=VALIDATION_SPLIT
)

embedding_layer = Embedding(MAX_NB_WORDS+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=False)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(LSTM_HIDDEN_SIZE))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(LSTM_HIDDEN_SIZE))(review_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print model.summary()
model.fit(x_train, y_train, validation_split=0.1,
          epochs=EPOCH, batch_size=BATCH_SIZE)



# building Hierachical Attention network
'''
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=EPOCH, batch_size=BATCH_SIZE)
'''

print 'saving model...'
# model.save(os.path.join(os.path.split(origin_path)[0], 'data/lstm_cnn.final'))
save_model(model, os.path.join(os.path.split(origin_path)[0], 'fusai_data/HATT.final'))

# model = load_model(os.path.join(os.path.split(origin_path)[0], 'fusai_data/HATT.final'))

re = model.predict(data, BATCH_SIZE)
id_index = np.array(data_val).flatten()
print len(re)
print len(id_index)
with open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'result.csv'), 'w') as f:
    for i, pre in enumerate(re):
        if pre[1] > 0.5:
            f.write(id_index[i] + ',' + 'POSITIVE' + '\n')
        else:
            f.write(id_index[i] + ',' + 'NEGATIVE' + '\n')