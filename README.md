# autocnn_unsup
Matlab scripts implementing the model from our work [`Autoconvolution for Unsupervised Feature Learning`] (http://arxiv.org/abs/1606.00611) on unsupervised layer-wise training of a convolution neural network based on recursive autoconvolution (AutoCNN).

We present scripts for MNIST ([autocnn_mnist.m] (autocnn_mnist.m)), CIFAR-10 ([autocnn_cifar10.m] (autocnn_cifar10.m)) 
and STL-10 ([autocnn_stl10.m] (autocnn_stl10.m)). 
Our slighly optimized scripts are much faster (see Tables below) than those used to produce results in the paper. 
Moreover, the classification results are also surprisingly improved.

## Example of running
```matlab
opts.matconvnet = 'home/code/3rd_party/matconvnet';
opts.vlfeat = 'home/code/3rd_party/vlfeat/toolbox/mex/mexa64';
opts.gtsvm = '/home/code/3rd_party/gtsvm/mex';
opts.n_folds = 10;
opts.n_train = 4000;
opts.arch = '1024c13-2p-conv0_4__128g-4ch-160c11-4p-conv2_3';
autocnn_cifar10(opts, 'augment', true)
```

## Requirements
For faster filters learning it's recommended to use [VLFeat] (http://www.vlfeat.org/), and for faster forward pass - [MatConvNet] (http://www.vlfeat.org/matconvnet/). Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either [GTSVM] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/) or [LIBSVM] (https://github.com/cjlin1/libsvm). Compared to LIBSVM, GTSVM is much faster (because of GPU) and implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). [LIBLINEAR] (https://github.com/cjlin1/liblinear) can also be used, but it shows worse performance compared to the RBF kernel.
If neither LIBSVM nor GTSVM is available, the code will use Matlab's [LDA] (http://www.mathworks.com/help/stats/fitcdiscr.html).

## Learning methods
Currently, the supported unsupervised learning methods are k-means, [convolutional k-means] (conv_kmeans.m), k-medoids, GMM, [ICA and ISA] (ica.m).
We use VLFeat's k-means to obtain our results.

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
- Xeon CPU E5-2620 v3 @ 2.40GHz

## Results
So far, the model is purely unsupervised, i.e. fine tuning is not applied. 
Also, no data augmentation and no cropping is applied, other than horizontal flipping when specified (see Tables below).
Flipping is not applied (so far) for test samples in any case.
We report 2 results (in table cells): with a single SVM / SVM committee.

### MNIST
Test error (%) on MNIST (100) with 100 labeled images per class and using all (60k) MNIST training data. 
In both cases we report average % for 10 random tests. SVM committees consist of 8 models in case of 1 layer 
and 11 models in case of 2 layers (see code for details). 

Will be updated soon.

Model | MNIST (100), single SVM / committee | MNIST, single SVM / committee
-------|:--------:|:--------:
256c13 | - / - | - / -
192c11-32g-64c9 | - / - | - / -


### CIFAR-10, STL-10
Test accuracy (%) on CIFAR-10 (400) with 400 labeled images per class and using all (50k) CIFAR-10 training data. 
In both cases we report average % for 10 random tests. SVM committees consist of 12 models in case of 1 layer 
and 16 models in case of 2 layers (see code for details).

Model | CIFAR-10 (400) | CIFAR-10
-------|:--------:|:--------:
1024c13 | 69.6 / 71.8 | 81.7 / 83.4
1024c13+flip | 72.4 / 74.6 | 83.5 / 85.0
420c13-128g-160c11 | 73.8 / 75.5 | 84.4 / 85.1
420c13-128g-160c11+flip | 75.9 / 77.4 | 85.8 / 86.4
675c13-256g-160c11 | 74.4 / 75.9 | 84.7 / 85.4
675c13-256g-160c11+flip | **76.4 / 77.9** | **86.0 / 86.6**

Approximate total (training+prediction) time for 1 test. We also report prediction time (required to process and 
classify all 10k test samples), which is more relevant in practice. 

Model | CIFAR-10 (400) | CIFAR-10 | CIFAR-10 (prediction)
-------|:--------:|:--------:|:--------:
1024c13 | **3 min / 3.5 min** | **4.5 min / 15 min** | **9 sec / 18 sec**
1024c13+flip | **3 min** / 4 min | 6 min / 25 min | **9 sec** / 25 sec
420c13-128g-160c11 | 15 min / 15.5 min | 30 min / 45 min | 3.1 min / 3.4 min
420c13-128g-160c11+flip | 16 min / 17 min | 45 min / 80 min | 3 min / 3.5 min
675c13-256g-160c11 | 21.5 min / 22 min | 50 min / 65 min | 5 min / 5.5 min
675c13-256g-160c11+flip | 24 min / 25 min | 80 min / 110 min | 5.5 min / 6 min

### STL-10

Test accuracy (%) on STL-10 using 10 predefined folds. SVM committees consist of 16 models in case of 1 layer 
and 20 models in case of 2 layers (see code for details). 

Will be updated soon.

Model | STL-10
-------|:--------:
1024c13 | - / -
1024c13+flip | - / -
420c13-128g-160c11 | - / -
420c13-128g-160c11+flip | - / -
675c13-256g-160c11 | - / -
675c13-256g-160c11+flip | - / -
