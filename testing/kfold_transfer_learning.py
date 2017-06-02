#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

"""
This module performs k-fold cross-validation using the Keras and scikit-learn
libraries. The module uses transfer learning and takes a set of features and 
their associated labels as input.

Please refer to the following papers if you use this script in your published
research:

    Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow.
    URL: https://github.com/fchollet/keras

    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
    O., Blondel, M. Prettenhofer, P. Weiss, R., Dubourg, V. Vanderplas, J., 
    Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011)
    Scikit-learn: Machine learning in Python. Journal of Machine Learning 
    Research (12), 2825–2830. 
    URL: http://www.jmlr.org/papers/v12/pedregosa11a.html

Performing k-fold cross-validation requires you to set a number of parameters, 
which are described below:

    input_f:       Path to the input file in HDF5 format, in which the features 
                   and labels are stored. The HDF5 groups are expected to be
                   named "features" and "labels".
    
    learning_rate: The initial learning rate used for training the network.
    
Usage:

    Run the module from the command line by using the following command:
    
    python kfold_transfer_learning.py    
"""

# Set the variables for training the model
input_f = 'data.h5'
learning_rate = 0.01

# Load features and labels from file
with h5py.File(input_f, 'r') as hf:
    data = hf["features"][:]
    labels = hf["labels"][:]

# Configure the validation parameters
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up a list for keeping scores
scores = []

# Loop over the folds
for n, (train, test) in enumerate(kfold.split(data, labels)):
    # Set up and compile the model
    model = Sequential()
    model.add(Flatten(input_shape=data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=1e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    # Split the data into training and testing sets
    traindata, trainlabels = data[train], labels[train]
    testdata, testlabels = data[test], labels[test]

    # Convert integer labels into one-hot encoded vectors
    trainlabels = np_utils.to_categorical(trainlabels, 3)
    testlabels = np_utils.to_categorical(testlabels, 3)

    # Start training the model
    training = model.fit(traindata, trainlabels, batch_size=32, shuffle=True,
                         nb_epoch=75, verbose=1,
                         validation_data=(testdata, testlabels))

    # Evaluate the model
    (loss, accuracy) = model.evaluate(testdata, testlabels, batch_size=32,
                                      verbose=1)

    # Save weights for each fold into a file
    model.save_weights('topnet-cv-fold_%s.h5' % (n+1),
                       overwrite=True)

    # Write the training history for each fold into a file
    with open('test_output/cv2/topnet-history-fold_%s.pkl' % (n+1), 'wb')\
            as histfile:
        pickle.dump(training.history, histfile)

    # Append accuracy to list of scores
    scores.append(accuracy)

# Print the scores and the best fold
print "%.4f (STDEV %.4f" % (np.mean(scores), np.std(scores))
print "Best result during epoch %s" % np.argmax(scores)
