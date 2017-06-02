#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tankbuster.cnn import CNNArchitecture

"""
This module performs k-fold cross-validation using the Keras and scikit-learn
libraries. The module uses data augmentation, generating images on the fly from
a set of original images.

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

    arch:          Neural network architecture, as defined in the tankbuster
                   library. This value must be "ConvNet", as ResNets are trained
                   and validated using a different module
                   (kfold_transfer_learning.py).
            
    input_dir:     Path to the root directory where the data is stored. Different
                   classes must be located in subdirectories under the root 
                   directory, e.g.
          
                   data/
                   data/class_1
                   data/class_2
    
    learning_rate: The initial learning rate used for training the network.

    opt:           Optimizer, either "SGD" or "RMSProp".
     
    target_size:   Target size of the images to be generated.
    
Usage:

    Run the module from the command line by using the following command:
    
    python kfold_data_aug.py
"""

# Set the variables for training the model
arch = "ConvNet"  # Architecture
input_dir = "testdata"  # Input directory
learning_rate = 0.001  # Learning rate
opt = "SGD"  # Optimizer
target_size = (150, 150)  # Target size for data augmentation

# Configure a callback to reduce the learning rate upon plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50,
                              cooldown=50, min_lr=0.0001, verbose=1)

# Path to pre-trained weights file, if used. Otherwise None.
weights = None

# Configure the validation parameters
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Configure TensorFlow session to allow GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target, config=config)

# Register TensorFlow session with Keras
K.set_session(sess)

# Set up a list for keeping validation scores
scores = []

# Read data & labels
data, labels, classes = [], [], {}

for (root, subdirs, files) in os.walk(input_dir):
    # Assign a numerical identifier to each class directory
    for i, class_dir in enumerate(subdirs):
        classes[class_dir] = i

    # Define allowed image extensions
    ext = ['png', 'jpg', 'jpeg']

    # Loop over the files in each directory
    for f in files:
        if f.split('.')[-1] in ext:  # Check file extension
            path = os.path.join(root, f)  # Get image path
            label = path.split('/')[-2]  # Extract class label from path
            numlabel = classes[label]  # Get numerical label from the dict

            print "*** Now processing {} / {} / {} ...".format(path,
                                                               label,
                                                               numlabel)

            # Load and preprocess image
            image = load_img(path, target_size=target_size)  # Load image
            features = img_to_array(image)  # Convert image to numpy array

            labels.append(numlabel)  # Append label to list
            data.append(features)  # Append features to list

# Convert data and labels to numpy arrays
data = np.asarray(data, dtype=np.float32)
labels = np.asarray(labels, dtype=np.float32)

# Initiate a TensorFlow session
with sess.as_default():
    for n, (train, test) in enumerate(kfold.split(data, labels)):

        # Select CNN architecture
        print "Setting up CNN architecture: {} ...".format(arch)
        model = CNNArchitecture.select(arch, target_size[0],
                                       target_size[1], 3, 3)

        # If selected, configure SGD optimizer
        if opt == "SGD":
            optimizer = SGD(lr=learning_rate, decay=1e-5,
                            momentum=0.9, nesterov=True)

        # If selected, configure RMSProp optimizer
        if opt == "RMSProp":
            optimizer = RMSprop(lr=learning_rate)

        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        # Split the data into training and testing sets
        traindata, trainlabels = data[train], labels[train]
        testdata, testlabels = data[test], labels[test]

        # Convert integer labels into one-hot encoded vectors
        trainlabels = np_utils.to_categorical(trainlabels, 3)
        testlabels = np_utils.to_categorical(testlabels, 3)

        # Check if pre-trained weights should be used
        if weights:
            print "Loading pre-trained weights from {} ...".format(weights)
            model.load_weights(weights)

        # Generate training data using gpu:0
        with tf.device('/gpu:0'):
            # Set up generator for training data
            training_generator = ImageDataGenerator(rescale=1./255,
                                                    rotation_range=10,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.05,
                                                    shear_range=0.2,
                                                    zoom_range=0.2,
                                                    horizontal_flip=True,
                                                    fill_mode='nearest'
                                                    )

            # Generate training data
            training_data = training_generator.flow(traindata,
                                                    trainlabels,
                                                    batch_size=256
                                                    )

        # Generate validation data using gpu:1
        with tf.device('/gpu:1'):
            # Set up generator for validation data
            validation_generator = ImageDataGenerator(rescale=1./255,
                                                      rotation_range=10,
                                                      width_shift_range=0.2,
                                                      height_shift_range=0.05,
                                                      shear_range=0.2,
                                                      zoom_range=0.2,
                                                      horizontal_flip=True,
                                                      fill_mode='nearest'
                                                      )

            # Generate validation data
            validation_data = validation_generator.flow(testdata,
                                                        testlabels,
                                                        batch_size=32
                                                        )

        # Start training the model
        training = model.fit_generator(training_data,
                                       steps_per_epoch=2048,
                                       epochs=100,
                                       validation_data=validation_data,
                                       validation_steps=256,
                                       callbacks=[reduce_lr]
                                       )

        # Evaluate the model
        (loss, accuracy) = model.evaluate(testdata,
                                          testlabels,
                                          batch_size=32,
                                          verbose=1)

        # Save weights for each fold into a file
        model.save_weights('{}-cv-fold_{}.h5'.format(arch, (n + 1)),
                           overwrite=True)

        # Write the training history for each fold into a file
        with open('{}-history-fold_{}.pkl'.format(arch, (n + 1)), 'wb') \
                as histfile:
            pickle.dump(training.history, histfile)

        # Append the accuracy to the list of scores
        scores.append(accuracy)

# Print the scores and the best fold
print "%.4f (STDEV %.4f)" % (np.mean(scores), np.std(scores))
print "Best result for fold %s" % np.argmax(scores)

