% Demo code for training and prediction on CIFAR-10 (or CIFAR-100) with a 1-2 layer AutoCNN
% The code can work without parameters and dependencies
% However, consider the following parameters to improve speed and classification accuracy:
% opts.matconvnet (optional, recommended) - path to the MatConvNet root directory, e.g, /home/code/3rd_party/matconvnet
% opts.vlfeat (optional, recommended) - path to the VLFeat mex directory, e.g, /home/code/3rd_party/vlfeat/toolbox/mex/mexa64
% opts.gtsvm (optional, recommended) - path to the GTSVM mex directory, e.g, /home/code/3rd_party/gtsvm/mex
% opts.libsvm (optional) - path to the LIBSVM root directory, e.g, /home/code/3rd_party/libsvm/matlab
%
% opts.n_train (optional) - number of labeled training samples (default: full test)
% opts.arch (optional) - network architecture (default: large 2 layer network)
% opts.dataDir (optional) - directory with CIFAR-10 (or CIFAR-100) data
% opts.cifar100 (optional) - true to run scripts on CIFAR-100
% opts can contain other parameters

function test_results = autocnn_cifar(varargin)

time_start = clock;
fprintf('\ntest %s on %s \n', upper('started'), datestr(time_start))

if (nargin == 0)
    opts = [];
elseif (isstruct(varargin{1}))
    opts = varargin{1};
end

if (~isfield(opts,'cifar100') || ~opts.cifar100)
    opts.cifar100 = false; % run tests on CIFAR-10
    opts.dataset_str = '10';
else
    opts.dataset_str = '100'; % run tests on CIFAR-100
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
    opts.arch = '1024c11-2p-conv0_3__128g-4ch-160c9-4p-conv2_3'; % define a large 2 layer architecture
end
sample_size = [32,32,3];
opts.net_init_fn = @() net_init(opts.arch, opts, 'sample_size', sample_size, varargin{:});
rootFolder = fileparts(mfilename('fullpath'));
if (~isfield(opts,'dataDir'))
    opts.dataDir = fullfile(rootFolder,sprintf('data/cifar%s',opts.dataset_str));
end
if (~exist(opts.dataDir,'dir'))
    mkdir(opts.dataDir)
end
fprintf('loading and preprocessing data \n')
opts.sample_size = sample_size;
[data_train, data_test] = load_CIFAR_data(opts);
opts.dataset = sprintf('cifar%s',opts.dataset_str);

if (~isfield(opts,'n_folds'))
    opts.n_folds = 1;
end

net = opts.net_init_fn();
% PCA dimensionalities (p_j) for the SVM committee
if (~isfield(opts,'PCA_dim'))
    if (numel(net.layers) > 1)
        opts.PCA_dim = [50:25:150,200:50:400,500:100:1000];
    else
        opts.PCA_dim = [50:25:150,200:50:400,500,600];
    end
end

for fold_id = 1:opts.n_folds
    
    opts.fold_id = fold_id;
    test_results = autocnn_unsup(data_train, data_test, net, opts);

    fprintf('test took %5.3f seconds \n', etime(clock,time_start));
    fprintf('test (fold %d/%d) %s on %s \n\n', fold_id, opts.n_folds, upper('finished'), datestr(clock))
    time_start = clock;
end

end

function [data_train, data_test] = load_CIFAR_data(opts)

if (opts.cifar100)
    unpackPath = fullfile(opts.dataDir, 'cifar-100-matlab');
    files = {'meta','train','test'};
else
    unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
    files = {'batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5',...
        'test_batch'};
end
if (any(cellfun(@(f) ~exist(fullfile(unpackPath,sprintf('%s.mat',f)),'file'),files)))
    % download and unpack CIFAR-10 (CIFAR-100)
    url = sprintf('http://www.cs.toronto.edu/~kriz/cifar-%s-matlab.tar.gz',opts.dataset_str);
    fprintf('downloading %s\n', url);
    untar(url, opts.dataDir);
end
opts.dataDir = unpackPath;
% load unwhitened training images anyway
if (opts.cifar100)
    imdb = load(fullfile(opts.dataDir,'train.mat'));
    data_train.images = imdb.data;
    data_train.labels = imdb.fine_labels;
else
    data_train = load(fullfile(opts.dataDir,'batches.meta.mat'));
    data_train.images = [];
    data_train.labels = [];
    for batch_id=1:5
        imdb = load(fullfile(opts.dataDir,sprintf('data_batch_%d',batch_id)));
        data_train.images = cat(1,data_train.images,imdb.data);
        data_train.labels = [data_train.labels;imdb.labels];
    end
end
% convert to the Matlab format
data_train.images = single(permute(reshape(data_train.images, [size(data_train.images,1),opts.sample_size]), [1,3,2,4]))./255;
data_train.images = reshape(data_train.images, [size(data_train.images,1),prod(opts.sample_size)]);
data_train.unlabeled_images = data_train.images; % unwhitened images (to learn filters and connections)

if (opts.whiten && exist(fullfile(opts.dataDir,'train_whitened.mat'),'file'))
    fprintf('loading whitened data \n')
    imdb = load(fullfile(opts.dataDir,'train_whitened'));
    data_train.images = imdb.data;
    data_train.labels = imdb.labels;
    imdb = load(fullfile(opts.dataDir,'test_whitened'));
    data_test.images = imdb.data;
    data_test.labels = imdb.labels;
else
    if (opts.cifar100)
        imdb = load(fullfile(opts.dataDir,'test'));
        imdb.labels = imdb.fine_labels;
    else
        imdb = load(fullfile(opts.dataDir,'test_batch'));
    end
    imdb.data = single(permute(reshape(imdb.data, [size(imdb.data,1),opts.sample_size]), [1,3,2,4]))./255;
    imdb.data = reshape(imdb.data, [size(imdb.data,1),prod(opts.sample_size)]);
    data_test.labels = imdb.labels;
    data_test.images = imdb.data; % unwhitened test images
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

% we use the first 4k-10k samples as unlabeled data, it's enough to learn filters and connections and perform PCA
unlabeled_ids = 1:4e3;
data_train.unlabeled_images = data_train.unlabeled_images(unlabeled_ids,:);
data_train.unlabeled_images_whitened = data_train.images(unlabeled_ids,:);

end