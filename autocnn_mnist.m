% Demo code for training and prediction on MNIST with a 1-2 layer AutoCNN
% The code can work without parameters and dependencies
% However, consider the following parameters to improve speed and classification accuracy:
% opts.matconvnet (optional, recommended) - path to the MatConvNet root directory, e.g, /home/code/3rd_party/matconvnet
% opts.vlfeat (optional, recommended) - path to the VLFeat mex directory, e.g, /home/code/3rd_party/vlfeat/toolbox/mex/mexa64
% opts.gtsvm (optional, recommended) - path to the GTSVM mex directory, e.g, /home/code/3rd_party/gtsvm/mex
% opts.libsvm (optional) - path to the LIBSVM root directory, e.g, /home/code/3rd_party/libsvm/matlab
%
% opts.n_train (optional) - number of labeled training samples (default: full test)
% opts.arch (optional) - network architecture (default: large 2 layer network)
% opts.dataDir (optional) - directory with MNIST data
% opts can contain other parameters

function test_results = autocnn_mnist(varargin)

time_start = clock;
fprintf('\ntest %s on %s \n', upper('started'), datestr(time_start))

if (nargin == 0)
    opts = [];
elseif (isstruct(varargin{1}))
    opts = varargin{1};
end

if (~isfield(opts,'whiten'))
    opts.whiten = false; % whitening is not applied
end
if (~isfield(opts,'batch_size'))
    opts.batch_size = 100;
end
if (~isfield(opts,'rectifier_param'))
    opts.rectifier_param = [0,Inf];
end
if (~isfield(opts,'rectifier'))
    opts.rectifier = {'abs','abs'};
end
if (~isfield(opts,'conv_norm'))
    opts.conv_norm = 'stat';
end
if (~isfield(opts,'arch'))
    opts.arch = '192c11-2p-conv1_3__32g-3ch-64c9-2p-conv2_3'; % define a 2 layer architecture
end

sample_size = [28,28,1];
opts.lcn_l2 = true; % local feature map normalization
opts.lcn = false; % LCN is turned off for MNIST
opts.net_init_fn = @() net_init(opts.arch, opts, 'sample_size', sample_size, varargin{:});
rootFolder = fileparts(mfilename('fullpath'));
if (~isfield(opts,'dataDir'))
    opts.dataDir = fullfile(rootFolder,'data/mnist');
end
if (~exist(opts.dataDir,'dir'))
    mkdir(opts.dataDir)
end
fprintf('loading and preprocessing data \n')
opts.sample_size = sample_size;
[data_train, data_test] = load_MNIST_data(opts);
opts.dataset = 'mnist';

if (~isfield(opts,'n_folds'))
    opts.n_folds = 1;
end

for fold_id = 1:opts.n_folds
    net = opts.net_init_fn(); % initialize a network
    % PCA dimensionalities (p_j) for the SVM committee
    if (~isfield(opts,'PCA_dim'))
        if (numel(net.layers) > 1)
            opts.PCA_dim = [50,70,90,100,120,150:50:400];
        else
            opts.PCA_dim = [50,70,90,100,120,150:50:250];
        end
    end
    opts.fold_id = fold_id;
    test_results = autocnn_unsup(data_train, data_test, net, opts);

    fprintf('test took %5.3f seconds \n', etime(clock,time_start));
    fprintf('test (fold %d/%d) %s on %s \n\n', fold_id, opts.n_folds, upper('finished'), datestr(clock))
    time_start = clock;
end

end

function [data_train, data_test] = load_MNIST_data(opts)
% adopted from the matconvnet example

files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'};

for i=1:numel(files)
    if (~exist(fullfile(opts.dataDir, files{i}), 'file'))
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, opts.dataDir) ;
    end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r');
x1=fread(f,inf,'uint8');
fclose(f);
data_train.images = reshape(permute(reshape(single(x1(17:end))./255,[opts.sample_size(1:2),60e3]),[3 2 1]),...
    [60e3,prod(opts.sample_size(1:2))]);
data_train.unlabeled_images = data_train.images;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f);
data_test.images = reshape(permute(reshape(single(x2(17:end))./255,[opts.sample_size(1:2),10e3]),[3 2 1]),...
    [10e3,prod(opts.sample_size(1:2))]);

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r');
y1=fread(f,inf,'uint8');
fclose(f);
data_train.labels = double(y1(9:end));

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r');
y2=fread(f,inf,'uint8');
fclose(f);
data_test.labels = double(y2(9:end));

if (opts.whiten)
    fprintf('performing data whitening \n')
    opts.pca_dim = [];
    opts.pca_epsilon = 0.05;
    opts.pca_mode = 'zcawhiten';
    [data_train.images, PCA_matrix, data_mean, L_regul] = pca_zca_whiten(data_train.images, opts);
    data_test.images = pca_zca_whiten(data_test.images, opts, PCA_matrix, data_mean, L_regul);
end

% we use the first 4k samples as unlabeled data, it's enough to learn filters and connections and perform PCA
unlabeled_ids = 1:4*10^3;
data_train.unlabeled_images = data_train.unlabeled_images(unlabeled_ids,:);
data_train.unlabeled_images_whitened = data_train.images(unlabeled_ids,:);

end