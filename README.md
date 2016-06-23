# autocnn_unsup
Matlab scripts implementing the model from our work [`Autoconvolution for Unsupervised Feature Learning`] (http://arxiv.org/abs/1606.00611) on unsupervised layer-wise training of a convolution neural network based on recursive autoconvolution (AutoCNN).

Currently, we present the scripts only for CIFAR-10. You can run the main demo script [autocnn_cifar10.m] (autocnn_cifar10.m). Our slighly optimized scripts are much faster (see Tables below) than those used to produce results in the paper. Moreover, the classification results are also surprisingly improved.
The updated results and timings, and the scripts for other datasets (MNIST, STL-10) will be uploaded.

## Requirements
For faster filters learning it's recommended to use [VLFeat] (http://www.vlfeat.org/), and for faster forward pass - [MatConvNet] (http://www.vlfeat.org/matconvnet/). Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either [GTSVM] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/) or [LIBSVM] (https://github.com/cjlin1/libsvm). Compared to LIBSVM, GTSVM is much faster (because of GPU) and implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). [LIBLINEAR] (https://github.com/cjlin1/liblinear) can also be used, but it shows worse performance compared to the RBF kernel.
If neither LIBSVM nor GTSVM is available, the code will use Matlab's [LDA] (http://www.mathworks.com/help/stats/fitcdiscr.html).

## Learning methods
Currently, the supported unsupervised learning methods are k-means, [convolutional k-means] (conv_kmeans.m), k-medoids and GMM, we use VLFeat's k-means to obtain our results.

## Testing Environment
- Ubuntu 16.04 LTS
- Matlab R2015b 
- CUDA 7.5 (installed via apt-get install nvidia-cuda-toolkit)
- [MatConvNet] (http://www.vlfeat.org/matconvnet/)
- [cuDNN-v5] (https://developer.nvidia.com/cudnn)
- [VLFeat] (http://www.vlfeat.org/)
- [GTSVM] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/)
- 64GB RAM
- NVIDIA GTX 980 Ti

## Results
Test accuracy (%) on CIFAR-10 (400) with 400 labeled images per class and using all CIFAR-10 training data. In both cases we report average % for 10 random tests. SVM committees consist of 12 models in case of 1 layer and 16 models in case of 2 layers (see code for details).
So far, the model is purely unsupervised, i.e. fine tuning is not applied. 
Also, no data augmentation and no cropping is applied, other than horizontal flipping when specified (see Tables below).

Model | CIFAR-10 (400), single SVM / committee | CIFAR-10, single SVM / committee
-------|:--------:|:--------:
1024c13 | 69.6 / 71.8 | 81.7 / 83.4
1024c13+flip | 72.4 / 74.6 | 83.5 / 85.0
420c13-128g-160c11 | 73.8 / 75.5 | 84.4 / 85.1
420c13-128g-160c11+flip | 75.9 / 77.4 | 85.8 / 86.4
675c13-256g-160c11 | 74.4 / 75.9 | 84.7 / 85.4
675c13-256g-160c11+flip | **76.4 / 77.9** | **86.0 / 86.6**

Approximate total (training+prediction) time for 1 test. We also report prediction time (required to process and classifty all 10k test samples, flipping is not applied for test samples), which is more relevant in practice. 

Model | CIFAR-10 (400), single SVM / committee | CIFAR-10, single SVM / committee | CIFAR-10 (prediction), single SVM / committee
-------|:--------:|:--------:|:--------:
1024c13 | **3 min / 3.5 min** | **4.5 min / 15 min** | 9 sec / 18 min
1024c13+flip | **3 min** / 4 min | 6 min / 25 min | 9 sec / 25 sec
420c13-128g-160c11 | 15 min / 15.5 min | 30 min / 45 min | 3.1 min / 3.4 min
420c13-128g-160c11+flip | 16 min / 17 min | 45 min / 80 min | 3 min / 3.5 min
675c13-256g-160c11 | 21.5 min / 22 min | 50 min / 65 min | 5 min / 5.5 min
675c13-256g-160c11+flip | 24 min / 25 min | 80 min / 110 min | 5.5 min / 6 min
