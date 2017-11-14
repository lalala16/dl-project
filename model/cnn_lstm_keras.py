import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model

from pre_process import load_data_keras

# embedding
embedding_size = 256       # word_embedding size
maxlen = 2048                # used to pad input tweet sequence
max_features = 50000       # vocabulary size

# cnn
kernel_size = 5
filters = 64
pool_size = 4

# lstm
lstm_output_size = 70

# dense
dense_size = 256            # optional, depends on performance

# training
batch_size = 32
epochs = 2


# loading training data
print 'loading data...'
x_train, y_train, x_test, y_test = load_data_keras.load_data(word_num_max=max_features, sequence_max=maxlen)


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

print 'compiling model...'
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],)

# creating some callbacks
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/lstm_cnn')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='checkpoint/lstm_cnn.{epoch:02d}.hdf5', period=1)

print 'training model...'
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[tensorboard_callback, checkpoint_callback],   # wait for specification
          validation_data=(x_test, y_test))

print 'saving model...'
model.save('model/lstm_cnn.final')