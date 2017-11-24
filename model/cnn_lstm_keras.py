import keras
import os
import sys
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import save_model,load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
from pre_process import load_data_keras
from utility.CsvUtility import CsvUtility

origin_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

# embedding
embedding_size = 128       # word_embedding size
maxlen = 800                # used to pad input tweet sequence
max_features = 7000      # vocabulary size

# cnn
kernel_size = 5
filters = 64
pool_size = 4

# lstm
lstm_output_size = 70

# dense
dense_size = 256            # optional, depends on performance

# training
batch_size = 256
epochs = 20

# gpu
use_gpu = False


# loading training data
print 'loading data...'
x_train, y_train, x_test, x_test_id = load_data_keras.load_data(word_num_max=max_features, sequence_max=maxlen)
# x_train, y_train, x_test, x_test_id = load_data_keras.reload_data()
# print x_train[5]
# print y_train[5]
# print x_test[3]
# print y_test[3]

# building model
print 'building model...'
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters=filters,
                 kernel_size=kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

if use_gpu:
    multi_model = multi_gpu_model(model, gpus=2)
    print 'compiling model...'
    multi_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],)

    # creating some callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn'))
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn.{epoch:02d}.hdf5'), period=1)

    print 'training model...'

    multi_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[tensorboard_callback, checkpoint_callback],   # wait for specification
              validation_split=0.2)
    print 'saving model...'
    save_model(multi_model, os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn.final'))
else:
    print 'compiling model...'
    model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'], )

    # creating some callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn'))
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn.{epoch:02d}.hdf5'), period=1)

    print 'training model...'

    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[tensorboard_callback, checkpoint_callback],  # wait for specification
                    validation_split=0.2)

    print 'saving model...'
    # model.save(os.path.join(os.path.split(origin_path)[0], 'data/lstm_cnn.final'))
    save_model(model, os.path.join(os.path.split(origin_path)[0], 'fusai_data/lstm_cnn.final'))

print 'predict data...'
re = model.predict(x_test, batch_size).ravel().tolist()
id_index = np.array(x_test_id).flatten()
# print id_index
with open(os.path.join(os.path.split(origin_path)[0] + '/fusai_data/', 'result.csv'), 'w') as f:
    for i, pre in enumerate(re):
        if pre>0.5:
            f.write(id_index[i] + ',' + 'POSITIVE' + '\n')
        else:
            f.write(id_index[i] + ',' + 'NEGATIVE' + '\n')
# CsvUtility.write_array_csv_test(re, os.path.split(origin_path)[0] + '/data/', 'result.csv')