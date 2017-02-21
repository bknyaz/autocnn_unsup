% Experiments on the MNIST test set

%% You need to set your own paths to respective dependencies
opts = [];
opts.vlfeat = '/home/boris/Project/3rd_party/vlfeat/toolbox/mex/mexa64';
opts.matconvnet = '/home/boris/Project/3rd_party/matconvnet';
liblinear_path = '/home/boris/Project/3rd_party/liblinear/matlab';
gtsvm_path = '/home/boris/Project/3rd_party/gtsvm/mex'; % for some experiments

% to solve the issue with locating vl_kmeans by Matlab (sometimes it happens)
addpath(opts.vlfeat)
cd /home/boris/Project/3rd_party/vlfeat/bin/glnxa64
vl_kmeans(rand(100,5), 2);
[d,~,~] = fileparts(mfilename('fullpath'));
cd(d)
vl_kmeans(rand(100,5), 2);
addpath(strcat(d,'/../'))

%% MNIST
opts.gtsvm = '';
opts.liblinear = liblinear_path;
opts.PCA_dim = [];

tests = {[1e3,10],[60e3,1]};
for test=tests
    opts.n_train = test{1}(1);
    opts.n_folds = test{1}(2);
    opts.arch = '128c11-4p-conv1_3' % RA
    autocnn_mnist(opts)
    opts.arch = '128c7-4p-conv0' % Raw
    autocnn_mnist(opts)
    opts.arch = '128c7-5p-3s-conv1_3__1g-128ch-1024c5-3p-conv0_2' % RA
    autocnn_mnist(opts)
    opts.arch = '128c7-5p-3s-conv0__1g-128ch-1024c5-3p-conv0' % Raw
    autocnn_mnist(opts)
end