#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils

"""
This module extracts features from a ResNet50 pre-trained on ImageNet and stores
them into a HDF5 file.

Please refer to the following papers if you use this script in your published
research:

    Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow.
    URL: https://github.com/fchollet/keras
    
    He, K., Zhang, X., Ren, S. and Sun, J. (2016) Deep residual learning for
    image recognition. In Proceedings of the IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR'16), pp. 770-778.

Arguments:

    -i/--input:  Path to the root directory containing the images that you wish
                 to extract features from. The images for each class should be
                 placed in subdirectories under the root directory, e.g.:
                    data/
                    data/class_1
                    data/class_2
                    
    -t/--target: Path to the file in which the extracted features should be
                 saved, e.g. extracted_features.h5

    -o/--onehot: Optional argument for using one-hot encoded labels, e.g. [0, 1]
                 instead of categorical labels, e.g. '0' or '1'.

Usage:

    Run the module from the command line by using the following command, e.g.:
    
    python extract_features.py -i data/ -t features.h5
"""

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define the arguments
ap.add_argument("-i", "--input", required=True,
                help="Path to the root directory containing the images that you"
                     "wish to extract features from, e.g. data/.")
ap.add_argument("-t", "--target", required=True,
                help="Path to the file in which the extracted features should"
                     "be saved, e.g. extracted_features.h5.")
ap.add_argument("-o", "--onehot", action='store_true', default=False)

# Parse the arguments
args = vars(ap.parse_args())

# Assign the arguments to variables
input_dir = args['input']
output_file = args['target']
onehot = args['onehot']

# Set up a dictionary for class identifiers
classes = {}

# Initialize ResNet50 with pre-trained weights to be used as feature extractor
model = ResNet50(include_top=False, weights='imagenet',
                 input_tensor=Input(shape=(224, 224, 3)))

# Compile the model and get 2048-d vector from the average pool layer
model = Model(inputs=model.input,
              outputs=model.get_layer('avg_pool').output)


# Define a function for extracting features
def extract_features(sourcedir):
    """
    This function parses a directory for images, extracts their labels, applies
    ImageNet preprocessing and extracts CNN codes from the final average pooling
    layer in ResNet50.

    Params:
        sourcedir: The root directory to parse for images.

    Returns:
        Feature vectors and labels.
    """
    data, labels = [], []

    for (root, subdirs, files) in os.walk(sourcedir):
        # Assign a numerical identifier to each class directory
        for i, class_dir in enumerate(subdirs):
            classes[class_dir] = i

        ext = ['png', 'jpg', 'jpeg']  # Define allowed image extensions

        # Loop over the files in each directory
        for f in files:
            if f.split('.')[-1] in ext:  # Check file extension
                path = os.path.join(root, f) # Get image path
                dirpath = os.path.dirname(path)  # Get directory name
                label = os.path.basename(dirpath)  # Get class label from path
                numlabel = classes[label]  # Get numerical label from the dict

                print("*** Now processing {} / {} / {} ...".format(
                    path, label, numlabel
                ))

                # Load and preprocess image
                image = load_img(path, target_size=(224, 224))  # Load image
                features = img_to_array(image)  # Convert image to numpy array

                # Expand image matrix dimensions for input
                features = np.expand_dims(features, axis=0)

                # Apply ImageNet preprocessing
                features = preprocess_input(features,
                                            data_format='channels_last')

                # Extract features
                features = model.predict(features, batch_size=1, verbose=0)

                labels.append(numlabel)  # Append label to list
                data.append(features)  # Append features to list

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # If requested, convert numerical labels into one-hot encoded vectors.
    if onehot:
        labels = np_utils.to_categorical(labels, len(classes))

    # Return data and labels
    return data, labels

# Extract features
features, labels = extract_features(input_dir)

# Open h5py file and save the features and labels
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset("features", data=features)
    hf.create_dataset("labels", data=labels)
