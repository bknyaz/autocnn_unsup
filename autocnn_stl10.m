% Demo code for training and prediction on STL-10 with a 1-2 layer AutoCNN
% The code can work without parameters and dependencies
% However, consider the following parameters to improve speed and classification accuracy:
% opts.matconvnet (optional, recommended) - path to the MatConvNet root directory, e.g, /home/code/3rd_party/matconvnet
% opts.vlfeat (optional, recommended) - path to the VLFeat mex directory, e.g, /home/code/3rd_party/vlfeat/toolbox/mex/mexa64
% opts.gtsvm (optional, recommended) - path to the GTSVM mex directory, e.g, /home/code/3rd_party/gtsvm/mex
% opts.libsvm (optional) - path to the LIBSVM root directory, e.g, /home/code/3rd_party/libsvm/matlab
%
% opts.n_train (optional) - number of labeled training samples (default: full test)
% opts.arch (optional) - network architecture (default: large 2 layer network)
% opts.dataDir (optional) - directory with STL-10 data
% opts can contain other parameters

function test_results = autocnn_stl10(varargin)

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
    opts.arch = '1024c21-4p-conv0_4__128g-4ch-160c13-8p-conv2_3'; % define a large 2 layer architecture
end
sample_size = [96,96,3];
opts.net_init_fn = @() net_init(opts.arch, opts, 'sample_size', sample_size, varargin{:});
rootFolder = fileparts(mfilename('fullpath'));
if (~isfield(opts,'dataDir'))
    opts.dataDir = fullfile(rootFolder,'data/stl10');
end
if (~exist(opts.dataDir,'dir'))
    mkdir(opts.dataDir)
end
fprintf('loading and preprocessing data \n')
opts.sample_size = sample_size;
[data_train, data_test] = load_STL10_data(opts, 0, 40e3);
opts.dataset = 'stl10';

if (~isfield(opts,'n_folds'))
    opts.n_folds = 1;
end

net = opts.net_init_fn(); % initialize a network
% PCA dimensionalities (p_j) for the SVM committee
if (~isfield(opts,'PCA_dim'))
    if (numel(net.layers) > 1)
        opts.PCA_dim = [40,70,80,120,150,250,300:100:1500];
    else
        opts.PCA_dim = [30:10:100,120,150:50:250,300:100:600];
    end
end

for fold_id = 1:opts.n_folds
    
    opts.fold_id = fold_id;
    data_train_fold = load_STL10_data(opts, fold_id, 0);
    data_train.images = data_train_fold.images;
    data_train.labels = data_train_fold.labels;
    test_results = autocnn_unsup(data_train, data_test, net, opts);
    % for STL-10 we learn filters and process test features only once and use them for all 10 training folds
    if (fold_id == 1), opts.test_features = test_results.test_features; net = test_results.net{1}; end
    
    fprintf('test took %5.3f seconds \n', etime(clock,time_start));
    fprintf('test (fold %d/%d) %s on %s \n\n', fold_id, opts.n_folds, upper('finished'), datestr(clock))
    time_start = clock;
end

end

function [data_train, data_test] = load_STL10_data(opts, fold_id, n_unlabeled)

data_train.unlabeled_images = [];
data_train.unlabeled_images_whitened = [];

% load unwhitened unlabeled images (to learn filters and connections) if requested
if (n_unlabeled)
    fprintf('loading unlabeled data \n')
    mFile = matfile(fullfile(opts.dataDir,'unlabeled.mat'));
    data_train.unlabeled_images = single(mFile.X(1:n_unlabeled,:))./255;
    data_train.unlabeled_images_whitened = data_train.unlabeled_images;
    clear mFile;
end

if (opts.whiten && exist(fullfile(opts.dataDir,'train_whitened.mat'),'file'))
    fprintf('loading whitened data \n')
    imdb = load(fullfile(opts.dataDir,'train_whitened'));
    data_train.label_names = imdb.class_names;
    if (fold_id <= 0)
        data_train.images = imdb.X;
        data_train.labels = imdb.y;
    else
        data_train.images = imdb.X(imdb.fold_indices{fold_id}, :);
        data_train.labels = imdb.y(imdb.fold_indices{fold_id});
    end
    % load whitened unlabeled images (to perform PCA) if requested
    if (n_unlabeled)
        fprintf('loading whitened unlabeled data \n')
        mFile = matfile(fullfile(opts.dataDir,'unlabeled_whitened.mat'));
        data_train.unlabeled_images_whitened = mFile.X(1:n_unlabeled,:);
        clear mFile;
    end
    % load test data if requested
    if (nargout > 1)
        imdb = load(fullfile(opts.dataDir,'test_whitened'));
        data_test.images = imdb.X;
        data_test.labels = imdb.y;
    end
else
    % load test data if requested
    if (nargout > 1)
        imdb_test = load(fullfile(opts.dataDir,'test'));
        data_test.images = single(imdb_test.X)./255;
        data_test.labels = imdb_test.y;
    end
    imdb = load(fullfile(opts.dataDir,'train'));
    data_train.label_names = imdb.class_names;
    data_train.images = single(imdb.X)./255;
    data_train.labels = imdb.y;
    if (opts.whiten)
        % we need for than 27k samples here, so load 40k
        fprintf('performing data whitening (this can take a long time) \n')
        opts.pca_dim = [];
        opts.pca_epsilon = 0.05;
        opts.pca_mode = 'zcawhiten';
        opts.pca_fast = false;
        whitened_data = opts;
        [X, whitened_data.PCA_matrix, whitened_data.data_mean, whitened_data.L_regul] = ...
            pca_zca_whiten(data_train.unlabeled_images, opts);
        save(fullfile(opts.dataDir,'whitening_matrix'),'-struct','whitened_data','-v7.3')
        save(fullfile(opts.dataDir,'unlabeled_whitened'),'X','opts','-v7.3')
        data_train.unlabeled_images_whitened = X;
        imdb.X = pca_zca_whiten(data_train.images, opts, whitened_data.PCA_matrix, whitened_data.data_mean, whitened_data.L_regul);
        data_train.images = imdb.X;
        save(fullfile(opts.dataDir,'train_whitened'),'-struct','imdb','-v7.3')
        if (nargout > 1)
            imdb_test.X = pca_zca_whiten(data_test.images, opts, whitened_data.PCA_matrix, whitened_data.data_mean, whitened_data.L_regul);
            data_test.images = imdb_test.X;
            save(fullfile(opts.dataDir,'test_whitened'),'-struct','imdb_test','-v7.3')
        end
    end
    if (fold_id > 0)
        data_train.images = data_train.images(imdb.fold_indices{fold_id}, :);
        data_train.labels = data_train.labels(imdb.fold_indices{fold_id});
    end
end

% we use the first 4k-10k samples as unlabeled data, it's enough to learn filters and connections and perform PCA
if (n_unlabeled)
    unlabeled_ids = 1:min(n_unlabeled,10e3);
    data_train.unlabeled_images = data_train.unlabeled_images(unlabeled_ids,:);
    data_train.unlabeled_images_whitened = data_train.unlabeled_images_whitened(unlabeled_ids,:);
end

end