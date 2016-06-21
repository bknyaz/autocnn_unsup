# autocnn_unsup
Matlab scripts implementing the model from the paper [`Autoconvolution for Unsupervised Feature Learning`] (http://arxiv.org/abs/1606.00611) on unsupervised layer-wise training of a convolution neural network based on recursive autoconvolution (AutoCNN)

Currently, we present the scripts only for CIFAR-10. You can run the main demo script [autocnn_cifar10.m] (autocnn_cifar10.m). Our slighly optimized scripts are much faster than those used to produce results in the paper. Moreover, the classification results are also surprisingly improved.
The updated results and timings, and the scripts for other datasets (MNIST, STL-10) will be uploaded later.

## Requirements
For faster filters learning it's recommended to use VLFeat (http://www.vlfeat.org/), and for faster forward pass - MatConvNet (http://www.vlfeat.org/matconvnet/). Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either ['GTSVM'] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/) or [LIBSVM] (https://github.com/cjlin1/libsvm). Compared to LIBSVM, GTSVM is much faster (because of GPU) and implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). [LIBLINEAR] (https://github.com/cjlin1/liblinear) can also be used, but it shows worse performance compared to the RBF kernel.


