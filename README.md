# autocnn_unsup
Matlab scripts implementing the model from "Recursive Autoconvolution for Unsupervised Learning of Convolutional Neural Networks" accepted to IJCNN-2017. The paper will be available soon.
There is the [`previous version of this paper`] (http://arxiv.org/abs/1606.00611).
There is also simple [Python code] (https://github.com/bknyaz/autocnn_unsup_py) to learn filters with recursive autoconvolution and k-means.

Scripts for MNIST ([autocnn_mnist.m] (autocnn_mnist.m)), CIFAR-10 (CIFAR-100) ([autocnn_cifar.m] (autocnn_cifar.m)) and STL-10 ([autocnn_stl10.m] (autocnn_stl10.m)) are available.

To reproduce results from the paper, run scripts from the [experiments](https://github.com/bknyaz/autocnn_unsup_v2/tree/master/experiments) folder.

## Example of running
```matlab
opts.matconvnet = 'your_path/matconvnet';
opts.vlfeat = 'your_path/vlfeat/toolbox/mex/mexa64';
opts.gtsvm = 'your_path/gtsvm/mex';
opts.n_folds = 10;
opts.n_train = 4000;
opts.PCA_dim = 1500;
opts.arch = '1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2';
autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true)
```
This script should lead to an average accuracy of about 79.4% on CIFAR-10 (400).


## Requirements
For faster filter learning it's recommended to use [VLFeat] (http://www.vlfeat.org/), and for faster forward 
pass - [MatConvNet] (http://www.vlfeat.org/matconvnet/). 
Although, in the scripts we try to make it possible to choose between built-in Matlab and third party implementations.

For classification it's required to install either [GTSVM] (http://ttic.uchicago.edu/~cotter/projects/gtsvm/),
[LIBLINEAR] (https://github.com/cjlin1/liblinear) or [LIBSVM] (https://github.com/cjlin1/libsvm).
Compared to LIBSVM, GTSVM is much faster (because of GPU) and 
implements a one-vs-all SVM classifier (which is usually better for datasets like CIFAR-10 and STL-10). 
[LIBLINEAR] (https://github.com/cjlin1/liblinear) shows worse performance compared to the RBF kernel available 
both in GTSVM and LIBSVM.
If neither of these is available, the code will use Matlab's [LDA] (http://www.mathworks.com/help/stats/fitcdiscr.html).

## Learning methods
Currently, the supported unsupervised learning methods are k-means, [convolutional k-means] (conv_kmeans.m), k-medoids, GMM, [PCA] (pca_zca_whiten.m), [ICA and ISA] (ica.m).
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
- The model is purely unsupervised, i.e., label information is not used to train filters.
- **flip** - flipping (horizontal reflection, mirroring) is applied both for training and test samples.
- **augment** - taking random crops, flipping, rotation and scaling.

A convolutional neural network (CNN) trained layer wise using unsupervised learning methods 
and recursive autoconvolution is abbreviated as AutoCNN.

### MNIST

Model                       | MNIST (100)         | MNIST         
-------                     |:--------:           |:--------:
AutoCNN-S1-128 + LinearSVM  | 2.45 &plusmn; 0.10  | 0.69
AutoCNN-S2 + LinearSVM      | 1.75 &plusmn; 0.10  | 0.39

AutoCNN-S1-128: `128c11-4p-conv1_3`

AutoCNN-S2: `128c7-5p-3s-conv1_3__1g-128ch-1024c5-3p-conv0_2`


### CIFAR-10 and CIFAR-100
Test accuracy (%) on CIFAR-10 (400) with 400 labeled images per class and using all (50k) training data of CIFAR-10 and CIFAR-100. 

Model                             | CIFAR-10 (400)      | CIFAR-10  | CIFAR-100
-------|:--------:|:--------:|:--------:
AutoCNN-L+**flip** + LinearSVM    | 77.6 &plusmn; 0.3   | 84.4      | -
AutoCNN-L32 + RBFSVM              | 76.4 &plusmn; 0.4   | 85.4      | 63.9
AutoCNN-L32+**flip** + RBFSVM     | 79.4 &plusmn; 0.3   | 87.9      | 67.8

AutoCNN-L: `256c5-3p-conv0_3__1g-256ch-1024c5-3p-conv0_3__1g-1024ch-2048c5-3p-conv0_2`

AutoCNN-L32: `1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2`


### STL-10

Average test accuracy (%) on STL-10 using 10 predefined folds.

Model                                   | STL-10            
-------|:--------:|:--------:
AutoCNN-L+**augment** + LinearSVM       | 73.1 &plusmn; 0.5     
AutoCNN-L32 + RBFSVM                    | 68.7 &plusmn; 0.5 
AutoCNN-L32+**augment** + RBFSVM        | 74.5 &plusmn; 0.6

AutoCNN-L: `256c7-4p-conv0_3__1g-256ch-1024c5-4p-conv0_3__1g-1024ch-2048c5-3p-conv0_2`

AutoCNN-L32: `1024c7-5p-conv0_3__32g-32ch-256c5-4p-conv0_2__32g-256ch-1024c5-3p-conv0_2`

AutoCNN-L32+**augment**: `1024c7-4p-conv0_3__32g-32ch-256c5-4p-conv0_2__32g-256ch-1024c5-3p-conv0_2`
