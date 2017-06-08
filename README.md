# Recognizing military vehicles in social media images using deep learning

This repository contains code associated with the conference paper "Recognizing military vehicles in social media images using deep learning", presented at the 2017 IEEE International Conference on Intelligence and Security informatics at Beijing, China in July 2017. The paper is available for download here: http://www.helsinki.fi/~thiippal/publications/2017-ieee-isi.pdf

The main purpose of this repository is to provide code related to developing the system described in the conference article. The actual system may be found here: https://github.com/thiippal/tankbuster

The repository is structured as follows.

| Directory | Description |
|:---|:---|
|<a href="https://github.com/DigitalGeographyLab/MilVehicles/tree/master/testing">/testing</a>|Code related to testing and evaluating neural networks|
|<a href="https://github.com/DigitalGeographyLab/MilVehicles/tree/master/training">/training</a>|Code related to training neural networks|

These scripts may be used to set up a pipeline for object recognition using two alternatives, data augmentation and transfer learning. The order in which the scripts should be run for either option are given below.

| Step | Data augmentation | Transfer learning |
|:---|:---|:---|
|1|random_search.py|extract_features.py|
|2|train_data_aug.py|random_search.py|
|3|kfold_data_aug.py|train_transfer_learning.py|
|4||kfold_transfer_learning.py|

## Required libraries

The system was developed using the following libraries. Make sure you have installed the version mentioned below or a newer version.

| Library | Version |
|:---|:---|
|<a href="https://www.h5py.org/">h5py</a>|2.7.0|
|<a href="https://keras.io">Keras</a>|2.0.2|
|<a href="https://python-pillow.org/">Pillow</a>|4.1.0|
|<a href="http://scikit-learn.org/">scikit-learn</a>|0.18.1|
|<a href="https://pypi.python.org/pypi/tankbuster/0.3.2">tankbuster</a>|0.3.2|
|<a href="https://www.tensorflow.org/">TensorFlow</a>|1.0.1|
|<a href="http://deeplearning.net/software/theano/">Theano</a>|0.9.0|

## Reference

Feel free to re-use any of the code in this repository. If the code benefits your published research, please consider citing this work using the following reference:

Hiippala, T. (2017) Recognizing military vehicles in social media images using deep learning. In *Proceedings of the 2017 IEEE International Conference on Intelligence and Security informatics*. July 22-24, Beijing, China.

Moreover, do not forget to cite the libraries that enable the research: the README.md file under each subdirectory of this repository lists the appropriate references.
