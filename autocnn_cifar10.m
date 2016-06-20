% Demo code for training and prediction on CIFAR-10 with a 1-2 layer AutoCNN
% The code can work without parameters and dependencies
% However, consider the following parameters to improve speed and classification accuracy:
% opts.matconvnet (optional, recommended) - path to the MatConvNet root directory, e.g, /home/code/3rd_party/matconvnet
% opts.vlfeat (optional, recommended) - path to the VLFeat mex directory, e.g, /home/code/3rd_party/vlfeat/toolbox/mex/mexa64
% opts.gtsvm (optional, recommended) - path to the GTSVM mex directory, e.g, /home/code/3rd_party/gtsvm/mex
% opts.libsvm (optional) - path to the LIBSVM root directory, e.g, /home/code/3rd_party/libsvm/matlab
%
% opts.n_train (optional) - number of labeled training samples (default: full test)
% opts.arch (optional) - network architecture (default: large 2 layer network)
% opts.dataDir (optional) - directory with CIFAR-10 data
% opts can contain other parameters

function test_results = autocnn_cifar10(varargin)

time_start = clock;
fprintf('\ntest %s on %s \n', upper('started'), datestr(time_start))

if (nargin == 0)
    opts = [];
elseif (isstruct(varargin{1}))
    opts = varargin{1};
end

if (~isfield(opts,'whiten'))
    opts.whiten = true; % whitening is applied
end
if (~isfield(opts,'batch_size'))
    opts.batch_size = 100;
end
if (~isfield(opts,'rectifier_param'))
    opts.rectifier_param = [0.25,25];
end
if (~isfield(opts,'rectifier'))
    opts.rectifier = {'relu','abs'};
end
if (~isfield(opts,'conv_norm'))
    opts.conv_norm = 'rootsift';
end
if (~isfield(opts,'arch'))
    opts.arch = '1024c13-2p-conv0_4__128g-4ch-160c11-4p-conv2_3'; % define a large 2 layer architecture
end
opts.net_init_fn = @() net_init(opts.arch, opts, 'sample_size', sample_size, varargin{:});
rootFolder = fileparts(mfilename('fullpath'));
if (~isfield(opts,'dataDir'))
    opts.dataDir = fullfile(rootFolder,'data/cifar10');
end
if (~exist(opts.dataDir,'dir'))
    mkdir(opts.dataDir)
end
fprintf('loading and preprocessing data \n')
[data_train, data_test] = load_CIFAR_data(opts);
opts.dataset = 'cifar10';

if (~isfield(opts,'n_folds'))
    opts.n_folds = 1;
end

for fold_id = 1:opts.n_folds
    net = opts.net_init_fn(); % initialize a network
    % PCA dimensionalities (p_j) for the SVM committee
    if (~isfield(opts,'PCA_dim'))
        if (numel(net.layers) > 1)
            opts.PCA_dim = [50:25:150,200:50:400,500:100:1000];
        else
            opts.PCA_dim = [50:25:150,200:50:400,500,600];
        end
    end
    opts.fold_id = fold_id;
    test_results = autocnn_unsup(data_train, data_test, net, opts);

    fprintf('test took %5.3f seconds \n', etime(clock,time_start));
    fprintf('test (fold %d/%d) %s on %s \n\n', fold_id, opts.n_folds, upper('finished'), datestr(clock))
    time_start = clock;
end

end

function [data_train, data_test] = load_CIFAR_data(opts)

unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
if (~exist(fullfile(unpackPath,'batches.meta.mat'),'file'))
    % download and unpack CIFAR-10
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
    fprintf('downloading %s\n', url);
    untar(url, opts.dataDir);
end
opts.dataDir = unpackPath;
% load unwhitened training images anyway
data_train = load(fullfile(opts.dataDir,'batches.meta.mat'));
data_train.images = [];
data_train.labels = [];
for batch_id=1:5
    imdb = load(fullfile(opts.dataDir,sprintf('data_batch_%d',batch_id)));
    n_samples = size(imdb.data,1);
    imdb.data = single(reshape(permute(reshape(imdb.data, [n_samples,opts.sample_size]), [1,3,2,4]), ...
        [n_samples,prod(opts.sample_size)]))./255;
    data_train.images = [data_train.images;imdb.data];
    data_train.labels = uint8([data_train.labels;imdb.labels]);
end
data_train.unlabeled_images = data_train.images; % unwhitened images (to learn filters and connections)

if (opts.whiten && exist(fullfile(opts.dataDir,'train_whitened.mat'),'file'))
    fprintf('loading whitened data \n')
    imdb = load(fullfile(opts.dataDir,'train_whitened'));
    data_train.images = imdb.data;
    data_train.labels = uint8(imdb.labels);
    imdb = load(fullfile(opts.dataDir,'test_whitened'));
    data_test.images = imdb.data;
    data_test.labels = uint8(imdb.labels);
else
    imdb = load(fullfile(opts.dataDir,'test_batch'));
    data_test.images = imdb.data; % unwhitened test images
    data_test.labels = uint8(imdb.labels);
    imdb.data = single(reshape(permute(reshape(imdb.data, [n_samples,opts.sample_size]), [1,3,2,4]), ...
        [n_samples,prod(opts.sample_size)]))./255;
    data_test.images = imdb.data;
    if (opts.whiten)
        fprintf('performing data whitening \n')
        opts.pca_dim = [];
        opts.pca_epsilon = 0.05;
        opts.pca_mode = 'zcawhiten';
        whitened_data = opts;
        [data, whitened_data.PCA_matrix, whitened_data.data_mean, whitened_data.L_regul] = ...
            pca_zca_whiten(data_train.unlabeled_images, opts);
        save(fullfile(opts.dataDir,'whitening_matrix'),'-struct','whitened_data','-v7.3')
        labels = data_train.labels;
        data_train.images = data;
        save(fullfile(opts.dataDir,'train_whitened'),'data','labels','-v7.3')
        data = pca_zca_whiten(data_test.images, opts, whitened_data.PCA_matrix, whitened_data.data_mean, whitened_data.L_regul);
        labels = data_test.labels;
        data_test.images = data;
        save(fullfile(opts.dataDir,'test_whitened'),'data','labels','-v7.3')
    end
end

% we use the first 4k samples as unlabeled data, it's enough to learn filters and connections and perform PCA
unlabeled_ids = 1:4*10^3;
data_train.unlabeled_images = data_train.unlabeled_images(unlabeled_ids,:);
data_train.unlabeled_images_whitened = data_train.images(unlabeled_ids,:);

end