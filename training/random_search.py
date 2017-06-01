#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, \
    Dense
from keras.regularizers import l2
from sklearn.model_selection import RandomizedSearchCV
from keras import backend as K
from keras.optimizers import SGD

"""
This module implements the random search proposed by Bergstra and Bengio (2012)
using the Keras and Scikit-learn libraries.

Please refer to the following papers if you use this script in your published
research:

    Bergstra, J. and Bengio, Y. (2012) Random search for hyper-parameter
    optimization. Journal of Machine Learning Research 13(1), 281–305.
    URL: http://www.jmlr.org/papers/v13/bergstra12a.html

    Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow.
    URL: https://github.com/fchollet/keras

    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
    O., Blondel, M. Prettenhofer, P. Weiss, R., Dubourg, V. Vanderplas, J., 
    Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011)
    Scikit-learn: Machine learning in Python. Journal of Machine Learning 
    Research (12), 2825–2830. 
    URL: http://www.jmlr.org/papers/v12/pedregosa11a.html

Performing a random search requires you to set a number of parameters, which are
described below:

    combinations: An integer defining the number of random parameter
                  combinations to try.
    data: Path to the root directory where the data is stored. Different classes
          must be located in subdirectories under the root directory, e.g.
          
                data/
                data/class_1
                data/class_2
                
    epochs: An integer defining for how many epochs the model is trained for.
    folds: An integer defining the number of folds trained for each parameter
           combination.
    target_size: A tuple defining the size of images to be fed to the network.

The actual parameters related to the model must be defined in the code. Here is
an example of parameter ranges, whose combinations will be tested during the
random search:

    batch_size = [32, 64, 128, 256]
    l2_lambda = [0.01, 0.001, 0.0001]
    dropout = [0.25, 0.5, 0.75]
    nodes = [64, 128, 256, 512]
    learning_rate = [0.1, 0.01, 0.001, 0.0001]
"""

# Define the parameters for running the search
combinations = 2
epochs = 10
data = 'testdata/'
folds = 2
target_size = (150, 150)

# Define lists of model parameters to be search
batch_size = [32, 64, 128, 256]
l2_lambda = [0.01, 0.001, 0.0001]
dropout = [0.25, 0.5, 0.75]
nodes = [64, 128, 256, 512]
learning_rate = [0.1, 0.01, 0.001, 0.0001]

# Construct a parameter dictionary incorporating the parameter lists
param_dict = dict(batch_size=batch_size, l2_lambda=l2_lambda, dropout=dropout,
                  nodes=nodes, learning_rate=learning_rate)

# Instruct Keras to use Theano backend (TensorFlow is default). Check that
# Theano backend is selected and set the correct image dimension ordering.
if K.backend() == 'theano':
    K.set_image_data_format('channels_first')

# Next, set up a dictionary for storing the class identifiers for the data.
# This dictionary is also used to define the number of nodes in the final layer
# of the neural network.
classes = {}


# Define a function for preprocessing data
def prepare_data(sourcedir):
    """
    This function parses a directory for images, extracts their labels,
    and converts them into numpy arrays.

    Params:
        sourcedir: A root directory with source images, which should be stored
                   in subdirectories, e.g.

                   root/
                   root/class_1
                   root/class_2

                   Both 'class_1' and 'class_2' would be then added to the
                   dictionary of classes defined before this function.

    Returns:
        Two NumPy arrays containing the images (as normalized NumPy arrays) and
        labels (as one-hot encoded vectors).
    """
    # Set up empty lists for storing the data and labels
    data, labels = [], []

    # Walk through the source directory
    for (root, subdirs, files) in os.walk(sourcedir):
        # Assign a numerical identifier to each class directory
        for i, class_dir in enumerate(subdirs):
            classes[class_dir] = i
            print("[INFO] Found class {}; "
                  "assigned identifier {}.".format(class_dir, i))

        # Define allowed image extensions
        ext = ['png', 'jpg', 'jpeg']

        # Loop over the files in each directory
        for f in files:
            # Check file extension
            if f.split('.')[-1] in ext:
                # Get image path
                path = os.path.join(root, f)
                # Extract class label from path
                label = path.split('/')[-2]
                # Get the corresponding label integer from the classes dict
                numlabel = classes[label]
                # Load image
                image = load_img(path, target_size=target_size)
                # Convert image to numpy array
                features = img_to_array(image)

                # Append data and labels to lists
                data.append(features)
                labels.append(numlabel)

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Convert numerical labels into one-hot encoded vectors
    labels = np_utils.to_categorical(labels, len(classes))

    # Normalize the RGB values into range 0...1
    data = data.astype('float') / 255.0

    # Return data and labels
    return data, labels

# Define a function that builds a neural network, which can be then passed to
# the scikit-learn API to perform the random search. Note that the
# hyperparameters to be searched need to be passed to the create_network
# function: dummy values will do, as these values are retrieved from the
# dictionary containing the hyperparameter lists.


def create_network(l2_lambda=0.0, dropout=0.0, learning_rate=0.1, nodes=0):
    # Initialize the model
    kmodel = Sequential()

    # Define the first convolutional block
    kmodel.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150),
                      kernel_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(Conv2D(32, (3, 3), kernel_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Define the second convolutional block
    kmodel.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))
    kmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    kmodel.add(Flatten())

    # Add dropout
    kmodel.add(Dropout(dropout))

    # Add a dense layer followed by an activation
    kmodel.add(Dense(nodes, kernel_regularizer=l2(l=l2_lambda)))
    kmodel.add(Activation("relu"))

    # Add dropout
    kmodel.add(Dropout(dropout))

    # Define the Softmax classifier
    kmodel.add(Dense(num_classes))
    kmodel.add(Activation("softmax"))

    # Define the optimizer
    sgd = SGD(lr=learning_rate)

    # Compile model
    kmodel.compile(loss="categorical_crossentropy", optimizer=sgd,
                   metrics=['categorical_accuracy'])

    return kmodel

# Prepare data
data, labels = prepare_data(data)

# Check the number of classes and assign the value to a variable. This value is
# used to define the number of neurons in the final Dense layer of the network.
num_classes = len(classes)

# Build Keras model
model = KerasClassifier(build_fn=create_network, verbose=1, epochs=epochs)

# Configure random search
rsearch = RandomizedSearchCV(estimator=model,
                             n_iter=combinations,
                             param_distributions=param_dict,
                             random_state=42,
                             scoring='neg_log_loss',
                             verbose=2,
                             cv=folds
                             )

# Perform the random search
rsearch.fit(data, labels)

# Print best result
print("Best: %.5f using %s" % (rsearch.best_score_,
                               rsearch.best_params_))

# Assign results to lists
mean = rsearch.cv_results_['mean_test_score']
std = rsearch.cv_results_['std_test_score']
param = rsearch.cv_results_['params']

# Loop over the results and print them:
if __name__ == '__main__':
    for m, s, p in zip(mean, std, param):
        print("Mean: %.5f (SD: %.5f) with %r" % (m, s, p))
