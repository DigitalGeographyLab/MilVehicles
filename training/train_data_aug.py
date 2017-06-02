#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau
from tankbuster.cnn import CNNArchitecture
from keras import backend as K
import tensorflow as tf

"""
This module trains a neural network from scratch using the Keras library. The 
module uses data augmentation, generating images on the fly from a set of 
original images.

Please refer to the following paper if you use this script in your published
research:

    Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow.
    URL: https://github.com/fchollet/keras

Performing k-fold cross-validation requires you to set a number of parameters, 
which are described below:

    arch:          Neural network architecture, as defined in the tankbuster
                   library. This value must be "ConvNet", as ResNets are trained
                   using a different module (train_transfer_learning.py).

    learning_rate: The initial learning rate used for training the network.

    opt:           Optimizer, either "SGD" or "RMSProp".

    target_size:   Target size of the images to be generated.
    
    train_dir:     Path to directory with training images. The images for each
                   class must be in their own subfolder, e.g.
                   
                   train/
                   train/class_1
                   train/class_2
                   
    val_dir:       Path to directory with validation images. The images for each
                   class must be in their own subfolder, e.g.
                   
                   val/
                   val/class_1
                   val/class_2

Usage:

    Run the module from the command line by using the following command:

    python train_data_aug.py
"""

# Set the variables for training the model
arch = "ConvNet"  # Architecture
train_dir = "train/"  # Directory containing training data
val_dir = "val/"  # Directory containing validation data
learning_rate = 0.01  # Learning rate
opt = "SGD"  # Optimizer
target_size = (150, 150)  # Target size for data augmentation

# Path to pre-trained weights file, if used. Otherwise None.
weights = None

# Configure a callback to reduce the learning rate upon plateau
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50,
                              cooldown=50, min_lr=0.00000001, verbose=1)

# Configure TensorFlow session to allow GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target, config=config)

# Register TensorFlow session with Keras
K.set_session(sess)

# Initiate TensorFlow session
with sess.as_default():

    # Select CNN architecture
    print "Setting up CNN architecture: {} ...".format(arch)
    model = CNNArchitecture.select(arch, target_size[0], target_size[1], 3, 3)

    # If selected, configure SGD optimizer
    if opt == "SGD":
        optimizer = SGD(lr=learning_rate, decay=1e-5,
                        momentum=0.9, nesterov=True)

    # If selected, configure RMSProp optimizer
    if opt == "RMSProp":
        optimizer = RMSprop(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    # Print model summary for log
    print model.summary()

    # Check if pre-trained weights should be used
    if weights:
        print "Loading pre-trained weights from {} ...".format(weights)
        model.load_weights(weights)

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

    # Generate training data
    training_data = training_generator.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=256,
    )

    # Generate validation data
    validation_data = validation_generator.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=32,
    )

    # Start training the model
    training = model.fit_generator(training_data,
                                   steps_per_epoch=2048,
                                   epochs=100,
                                   validation_data=validation_data,
                                   validation_steps=256,
                                   callbacks=[reduce_lr]
                                   )

    # Create timestamp for filename
    stamp = str(datetime.now()).split(' ')[0]

    print "*** Saved model, weights and history to file ..."
    # Save weights into file
    model.save_weights('weights-{}-{}-{}-aug.h5'.format(arch,
                                                        stamp,
                                                        learning_rate),
                       overwrite=True)

    # Write the training history to file
    with open('history_{}-{}-{}.pkl'.format(arch,
                                            stamp,
                                            learning_rate),
              'wb') as hfile:

        # Dump model history dictionary into the file
        pickle.dump(training.history, hfile)
