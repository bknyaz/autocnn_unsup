## Update
This version of README is kept for reference. To reproduce these results, use the following commit [f8f94e1](https://github.com/bknyaz/autocnn_unsup/commit/f8f94e1c7135d706561a630f151ca0af478b864b).

A fresh version of [README](README.md).

# autocnn_unsup
Matlab scripts implementing the model from our work 
[`Autoconvolution for Unsupervised Feature Learning`](https://arxiv.org/abs/1606.00611v1) 
on unsupervised layer-wise training of a convolution neural network based on recursive autoconvolution (AutoCNN).

We present scripts for MNIST ([autocnn_mnist.m](autocnn_mnist.m)), CIFAR-10 (CIFAR-100) ([autocnn_cifar.m](autocnn_cifar.m)) 
and STL-10 ([autocnn_stl10.m](autocnn_stl10.m)). 
Our slighly optimized scripts are much faster (see Tables below) than those used to produce results in the paper. 
Moreover, the classification results are also a little bit improved.

## Example of running
```matlab
opts.matconvnet = 'home/code/3rd_party/matconvnet';
opts.vlfeat = 'home/code/3rd_party/vlfeat/toolbox/mex/mexa64';
opts.gtsvm = '/home/code/3rd_party/gtsvm/mex';
opts.n_folds = 10;
opts.n_train = 4000;
opts.arch = '1024c11-2p-conv0_4__256g-4ch-160c9-4p-conv2_3';
autocnn_cifar(opts, 'augment', true)
```
This script should obtain an average accuracy of about 78.5% on CIFAR-10 (400).

## Requirements
For faster filter learning it's recommended to use [VLFeat](http://www.vlfeat.org/), and for faster forward 
pass - [MatConvNet](http://www.vlfeat.org/matconvnet/). 
Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either [GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/),
[LIBLINEAR](https://github.com/cjlin1/liblinear) or [LIBSVM](https://github.com/cjlin1/libsvm).
Compared to LIBSVM, GTSVM is much faster (because of GPU) and 
implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). 
[LIBLINEAR](https://github.com/cjlin1/liblinear) shows worse performance compared to the RBF kernel available 
both in GTSVM and LIBSVM.
If neither of these is available, the code will use Matlab's [LDA](http://www.mathworks.com/help/stats/fitcdiscr.html).

## Learning methods
Currently, the supported unsupervised learning methods are k-means, [convolutional k-means](conv_kmeans.m), k-medoids, GMM, PCA, [ICA and ISA](ica.m).
We use VLFeat's k-means to obtain our results.

## Testing environment
- Ubuntu 16.04 LTS
- Matlab R2015b 
- CUDA 7.5 (installed via apt-get install nvidia-cuda-toolkit)
- [MatConvNet](http://www.vlfeat.org/matconvnet/)
- [cuDNN-v5](https://developer.nvidia.com/cudnn)
- [VLFeat](http://www.vlfeat.org/)
- [GTSVM](http://ttic.uchicago.edu/~cotter/projects/gtsvm/)
- 64GB RAM
- NVIDIA GTX 980 Ti
- Xeon CPU E5-2620 v3 @ 2.40GHz

## Results
- So far, the model is purely unsupervised, i.e., label information is not used to train filters.
- Also, no data augmentation and no cropping is applied, other than horizontal flipping when specified (see Tables below).
- **flip** - indicates that flipping (horizontal reflection, mirroring) is applied both for training and test samples.
- We report 2 results (in table cells): with a single SVM / SVM committee.
- For 2 layers, an average number of filters used after prunning is indicated in layer 1 
(i.e., 90 instead of 192, or 650 instead of 1024), see the paper for details.

We have minor fixes in code, so the results are updated according to the current version of the code.

### MNIST
Test error (%) on MNIST (100), MNIST (300) with 100 or 300 labeled images per class, and using all (60k) MNIST training data (full test).
In both cases we report average % for 10 tests.
SVM committees consist of 7-11 models (see code for details). 
In our paper, the results on MNIST were obtained using LIBSVM. Here, we use GTSVM.

Model           | MNIST (100)   |MNIST (300)    | MNIST         | MNIST (total time for 1 full test)
-------         |:--------:     |:--------:     |:--------:     |:--------:
256c13          | 2.15 / 2.00   | 1.41 / 1.31   | 0.46 / 0.45   | 1 min / 2 min
90c11-32g-64c9  | 1.58 / 1.54   | 0.94 / 0.93   | 0.40 / 0.39   | 6 min / 8.5 min

For MNIST we observe a very large variance of the classification error.
Among 10 runs on full MNIST, our minimum error with a single SVM (PCA = 350) was **0.33%**, with an SVM committee - **0.34%**. 

Full definitions of architectures are following:

1 layer: `256c13-4p-conv1_3`

2 layers: `192c11-2p-conv1_3__32g-3ch-64c9-2p-conv2_3`

### CIFAR-10
Test accuracy (%) on CIFAR-10 (400) with 400 labeled images per class and using all (50k) CIFAR-10 training data. 
In all cases we report average % for 10 tests unless otherwise specified.
SVM committees consist of 9-21 models (see code for details).

Model                       | CIFAR-10 (400)    | CIFAR-10
-------|:--------:|:--------:
1024c13                     | 69.6 / 71.8       | 81.7 / 83.4
1024c13+**flip**            | 72.6 / 74.6       | 84.1 / 85.3
650c11-256g-160c9           | 74.2 / 76.4       | 84.8 / 85.6       (1 test)
650c11-256g-160c9+**flip**  | **76.9 / 78.5**   | **86.9 / 87.3**   (1 test)

Full definitions of architectures are following:

1 layer: `1024c13-8p-conv0_4`

2 layers: `1024c11-2p-conv0_3__Ng-4ch-160c9-4p-conv2_3`

##### Timings
Approximate total (training+prediction) time for 1 test. 
We also report prediction time required to process and classify all 10k test samples. 

Model                       | CIFAR-10 (400)        | CIFAR-10              | CIFAR-10 (prediction)
-------|:--------:|:--------:|:--------:
1024c13                     | **3 min / 3.5 min**   | **4.5 min / 15 min**  | **9 sec / 18 sec**
1024c13++**flip**           | **3 min** / 4 min     | 6 min / 25 min        | 17 sec / 48 sec
650c11-256g-160c9           | 29 min / 30 min       | 40 min / 70 min       | 3.7 min / 4.2 min
650c11-256g-160c9+**flip**  | 30 min / 31.5 min     | 65 min / 130 min      | 7.5 min / 9 min

Our SVM committee is several times cheaper computationally compared to a more traditional form of a committee 
(i.e., when a model is trained from scratch several times).


##### Learned filters and connections

Filters and connections are learned with architecture `opts.arch = '256c11-2p-conv0_3__64g-3ch-128c9-4p-conv2_3'`.
Filters are sorted according to their joint spatial and frequency resolution.

256 filters learned with k-means and conv_orders = [0:4] in layer 1 (*left*);
same, buth with l2-norm before k-means (*right*)

![conv0_4_layer1_kmeans_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv0_4_layer1_kmeans_cifar10.png)
![conv0_4_layer1_kmeans_l2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv0_4_layer1_kmeans_l2_cifar10.png)

64 connections from layer 1 to layer 2 visualized as the filters of layer 1 (on the left above) connected into 64 groups of 3

![connections_layer1_2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/connections_layer1_2_cifar10.png)

128 filters learned with k-means and conv_orders = [2:3] in layer 2 in case of 3 channels per feature map group (*left*);
same, buth with l2-norm before k-means (*right*)

![conv2_3_layer2_kmeans_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv2_3_layer2_kmeans_cifar10.png)
![conv2_3_layer2_kmeans_l2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv2_3_layer2_kmeans_l2_cifar10.png)

For classification, the filters on the left are better.

### CIFAR-100
Average test accuracy (%) on CIFAR-100 for 10 tests.
All model settings are identical to CIFAR-10.

Model                       | CIFAR-100     			| CIFAR-100 (total time for 1 full test)
-------|:--------:|:--------:
1024c13                     | 56.5 / 59.6               | 10 min / 75 min
1024c13+**flip**            | 60.3 / 62.7   			| 14 min / 120 min
650c11-256g-160c9           | 61.8 / 64.1 (1 test)		| 45 min / 160 min
650c11-256g-160c9+**flip**  | **65.9 / 67.1** (1 test)  | 70 min / 300 min


### STL-10

Average test accuracy (%) on STL-10 using 10 predefined folds.
SVM committees consist of 16 models in case of 1 layer and 19 models in case of 2 layers (see code for details).

Model                           | STL-10            | STL-10 (total time for 10 folds)
-------|:--------:|:--------:
1024c29                         | 60.0 / 62.8       | 32 min / 34 min
1024c29+**flip**                | 64.1 / 66.1       | 43 min / 46 min
670c21-256g-160c13              | 66.7 / 69.8       | 57 min / 63 min
670c21-256g-160c13+**flip**     | **70.8 / 72.6**   | 75 min / 85 min

Full definitions of architectures are following:

1 layer: `1024c29-20p-conv0_4`

2 layers: `1024c21-4p-conv0_4__Ng-4ch-160c13-8p-conv2_3`
