# Training

This folder includes code related to training the neural networks. The individual files are described in the table below. The **References** column lists the publications you should cite if you use the scripts in your published research.

| File | Description | References|
|:---|:---|:---|
|extract_features.py|A script for extracting features from images using a ResNet50 pre-trained on ImageNet.|1, 2|
|random_search.py|A script for conducting a random search for optimal hyperparameters.|1, 2, 3, 4|
|train_data_aug.py|A script for training a neural network from scratch using data augmentation.|1|



## References

1. Chollet, F. (2015-) Keras: Deep learning library for Theano and Tensorflow. URL: https://github.com/fchollet/keras

2. He, K., Zhang, X., Ren, S. and Sun, J. (2016) Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR'16)*, pp. 770–778.

3. Bergstra, J. and Bengio, Y. (2012) Random search for hyper-parameter optimization. *Journal of Machine Learning Research* 13(1), 281–305. URL: http://www.jmlr.org/papers/v13/bergstra12a.html

4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M. Prettenhofer, P., Weiss, R., Dubourg, V. Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011) scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* (12), 2825–2830. URL: http://www.jmlr.org/papers/v12/pedregosa11a.html