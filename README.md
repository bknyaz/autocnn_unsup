# autocnn_unsup
Matlab scripts implementing the model from our work 
[`Autoconvolution for Unsupervised Feature Learning`] (http://arxiv.org/abs/1606.00611) 
on unsupervised layer-wise training of a convolution neural network based on recursive autoconvolution (AutoCNN).

We present scripts for MNIST ([autocnn_mnist.m] (autocnn_mnist.m)), CIFAR-10 (CIFAR-100) ([autocnn_cifar.m] (autocnn_cifar.m)) 
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
opts.arch = '1024c11-2p-conv0_4__128g-4ch-160c9-4p-conv2_3';
autocnn_cifar(opts, 'augment', true)
```
This scripts should obtain an average accuracy of about 77.4% on CIFAR-10 (400).

## Requirements
For faster filters learning it's recommended to use [VLFeat] (http://www.vlfeat.org/), and for faster forward 
pass - [MatConvNet] (http://www.vlfeat.org/matconvnet/). 
Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either [GTSVM] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/) 
or [LIBSVM] (https://github.com/cjlin1/libsvm). Compared to LIBSVM, GTSVM is much faster (because of GPU) and 
implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). 
[LIBLINEAR] (https://github.com/cjlin1/liblinear) can also be used, but it shows worse performance compared to 
the RBF kernel.
If neither LIBSVM nor GTSVM is available, the code will use Matlab's [LDA] (http://www.mathworks.com/help/stats/fitcdiscr.html).

## Learning methods
Currently, the supported unsupervised learning methods are k-means, [convolutional k-means] (conv_kmeans.m), k-medoids, GMM, [ICA and ISA] (ica.m).
We use VLFeat's k-means to obtain our results.

## Testing environment
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
- So far, the model is purely unsupervised, i.e., label information is not used to train filters
- Also, no data augmentation and no cropping is applied, other than horizontal flipping when specified (see Tables below).
- flip - indicates that flipping (horizontal reflection, mirroring) is applied only for training samples 
(the test set is not augmented).
- **flip** - indicates that flipping is applied both for training and test samples.
- We report 2 results (in table cells): with a single SVM / SVM committee.
- For 2 layers, an average number of filters used after prunning is indicated in layer 1 
(i.e., 90 instead of 192, or 675 instead of 1024), see the paper for details.

### MNIST
Test error (%) on MNIST (100), MNIST (300) with 100 or 300 labeled images per class, and using all (60k) MNIST 
training data (full test).
In both cases we report average % for 10 random tests. 
SVM committees consist of 8 models in case of 1 layer and 11 models in case of 2 layers (see code for details). 
In our paper, the results on MNIST were obtained using LIBSVM. Here, we use GTSVM.

Model           | MNIST (100)   |MNIST (300)    | MNIST         | MNIST (total time for 1 full test)
-------         |:--------:     |:--------:     |:--------:     |:--------:
256c13          | 2.15 / 2.00   | 1.41 / 1.31   | 0.46 / 0.45   | 1 min / 2 min
90c11-32g-64c9  | 1.51 / 1.49   | 1.02 / 0.96   | 0.39 / 0.40   | 7 min / 10 min

For MNIST we observe a very large variance of the classification error.
Among 10 runs on full MNIST, our minimum error with a single SVM (PCA = 300) was 0.35%, with an SVM committee - 0.34%. 

Full definitions of architectures are following:

1 layer: `256c13-4p-conv1_3`

2 layers: `192c11-2p-conv1_3__32g-3ch-64c9-2p-conv2_3`

### CIFAR-10
Test accuracy (%) on CIFAR-10 (400) with 400 labeled images per class and using all (50k) CIFAR-10 training data. 
In both cases we report average % for 10 random tests. 
SVM committees consist of 12 models in case of 1 layer and 16 models in case of 2 layers (see code for details).

Model                       | CIFAR-10 (400)    | CIFAR-10
-------|:--------:|:--------:
1024c13                     | 69.6 / 71.8       | 81.7 / 83.4
1024c13+flip                | 72.4 / 74.6       | 83.5 / 85.0
420c13-128g-160c11          | 73.8 / 75.5       | 84.4 / 85.1
420c13-128g-160c11+flip     | 75.9 / 77.4       | 85.8 / 86.4
675c13-256g-160c11          | 74.4 / 75.9       | 84.7 / 85.4
675c13-256g-160c11+flip     | 76.4 / 77.9       | 86.0 / 86.6
650c11-256g-160c9           | 74.6 / 76.1       | 84.7 / 85.4
650c11-256g-160c9+**flip**  | **77.1 / 78.2**   | **86.6 / 87.1**

Full definitions of architectures are following:

1 layer: `1024c13-8p-conv0_4`

2 layers: `1024c11-2p-conv0_3__Ng-4ch-160c9-4p-conv2_3`

##### Timings
Approximate total (training+prediction) time for 1 test. 
We also report prediction time required to process and classify all 10k test samples. 

Model                       | CIFAR-10 (400)        | CIFAR-10              | CIFAR-10 (prediction)
-------|:--------:|:--------:|:--------:
1024c13                     | **3 min / 3.5 min**   | **4.5 min / 15 min**  | **9 sec / 18 sec**
1024c13+flip                | **3 min** / 4 min     | 6 min / 25 min        | **9 sec** / 25 sec
420c13-128g-160c11          | 15 min / 15.5 min     | 30 min / 45 min       | 3.1 min / 3.4 min
420c13-128g-160c11+flip     | 16 min / 17 min       | 45 min / 80 min       | 3 min / 3.5 min
675c13-256g-160c11          | 21.5 min / 22 min     | 50 min / 65 min       | 5 min / 5.5 min
675c13-256g-160c11+flip     | 24 min / 25 min       | 80 min / 110 min      | 5.5 min / 6 min
650c11-256g-160c9           | 19 min / 19.5 min     | 36 min / 52 min       | 3.4 min / 3.7 min
650c11-256g-160c9+**flip**  | 24 min / 25 min       | 55 min / 90 min       | 7 min / 8 min

Our SVM committee is several times cheaper computationally compared to a more traditional form of a committee 
(i.e., when a model is trained from scratch several times).

##### Learned filters

Filters and connections are learned with architecture opts.arch = '256c11-2p-conv0_3__64g-3ch-128c9-4p-conv2_3'.
Filters are sorted according to their joint spatial and frequency resolution.

256 filters learned with k-means and conv_orders = [0:4] in layer 1
256 filters learned with k-means and conv_orders = [0:4] in layer 1, 
l2-normalization is applied before k-means

![conv0_4_layer1_kmeans_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv0_4_layer1_kmeans_cifar10.png)
![conv0_4_layer1_kmeans_l2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv0_4_layer1_kmeans_l2_cifar10.png)

64 connections from layer 1 to layer 2 visualized as the filters of layer 1 connected into 64 groups of 3

![connections_layer1_2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/connections_layer1_2_cifar10.png)

128 filters learned with k-means and conv_orders = [2:3] in layer 2 in case of 3 channels per feature map group

![conv2_3_layer2_kmeans_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv2_3_layer2_kmeans_cifar10.png)

128 filters learned with k-means and conv_orders = [2:3] in layer 2 in case of 3 channels per feature map group, 
l2-normalization is applied before k-means

![conv2_3_layer2_kmeans_l2_cifar10](https://raw.githubusercontent.com/bknyaz/autocnn_unsup/master/figs/conv2_3_layer2_kmeans_l2_cifar10.png)

### CIFAR-100
All model settings are identical to CIFAR-10.

Model                       | CIFAR-100
-------|:--------:
1024c13                     | - / -
1024c13+**flip**            | - / -
650c11-256g-160c9           | - / - 
650c11-256g-160c9+**flip**  | - / - 


### STL-10

Average test accuracy (%) on STL-10 using 10 predefined folds. 
SVM committees consist of 16 models in case of 1 layer and 19 models in case of 2 layers (see code for details). 

Model                           | STL-10            | STL-10 (total time for 10 folds)
-------|:--------:|:--------:
1024c29                         | 60.0 / 62.8       | 32 min / 34 min
1024c29+**flip**                | 64.1 / 66.1       | 43 min / 46 min
420c21-128g-160c13              | 66.0 / 69.0       | 46 min / 49 min
420c21-128g-160c13+**flip**     | 69.8 / 71.8       | 50 min / 60 min
675c21-256g-160c13              | 66.1 / 69.1       | 64 min / 65 min
675c21-256g-160c13+**flip**     | **70.6 / 72.3**   | 80 min / 90 min

Full definitions of architectures are following:

1 layer: `1024c29-20p-conv0_4`

2 layers: `1024c21-4p-conv0_4__Ng-4ch-160c13-8p-conv2_3`


