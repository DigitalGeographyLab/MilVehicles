#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import h5py
from datetime import datetime
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

"""
This trains a neural network using the Keras library. The module uses transfer 
learning and takes a set of features and their associated labels as input.

Please refer to the following papers if you use this script in your published
research:

    Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow.
    URL: https://github.com/fchollet/keras

Performing transfer learning requires you to set a number of parameters, which 
are described below:

    input_f:       Path to the input file in HDF5 format, in which the features 
                   and labels are stored. The HDF5 groups are expected to be
                   named "features" and "labels".

    learning_rate: The initial learning rate used for training the network.

Usage:

    Run the module from the command line by using the following command:

    python train_transfer_learning.py    
"""

# Set the variables for training the model
input_f = 'data.h5'
learning_rate = 0.01

# Load features and labels from file
with h5py.File(input_f, 'r') as hf:
    data = hf["features"][:]
    labels = hf["labels"][:]

# Configure and compile the model
model = Sequential()
model.add(Flatten(input_shape=data.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
optimizer = SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

training = model.fit(data, labels, validation_split=0.1, batch_size=32,
                     shuffle=True, epochs=250, verbose=1)

# Create timestamp for filename
stamp = str(datetime.now()).split(' ')[0]

print "*** Saved weights and history to file ..."
# Save weights to file
model.save_weights('test_output/weights-{}-{}-tl.h5'.format(stamp,
                                                            learning_rate),
                   overwrite=True)

# Write training history to file
with open('test_output/history_{}_{}-tl-gen.pkl'.format(stamp,
                                                        learning_rate),
          'wb') as hfile:

    # Dump model history dictionary into the file
    pickle.dump(training.history, hfile)
