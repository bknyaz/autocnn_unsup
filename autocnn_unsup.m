function test_results = autocnn_unsup(data_train, data_test, net, opts)
% AutoCNN main pipeline suitable for image datasets such as MNIST, CIFAR, STL-10

fprintf('setting up GPU and %s \n', upper('dependencies'))
[net, opts] = set_up(net, opts); % check GPU, dependencies and add paths
net.layers{1}.flip = false;
opts.norm = 'stat';

if (~isfield(opts,'n_train'))
    opts.n_train = size(data_train.images,1); % full test
end
labels = unique(data_train.labels);
n_classes = length(labels);
% get training samples ids for reduced datasets: CIFAR-10 (400), MNIST (100), etc.
if (opts.n_train >= size(data_train.images,1))
    train_ids = 1:size(data_train.images,1);
else
    train_ids = zeros(opts.n_train,1);
    n = opts.n_train/n_classes; % number of training samples per class
    for k=1:n_classes
        ids = find(data_train.labels == labels(k));
        train_ids((k-1)*n+1:k*n) = ids(randperm(length(ids),n));
    end
end
train_ids = train_ids(randperm(length(train_ids)));
n = min([1000,size(data_train.unlabeled_images,1),length(train_ids)]);
data_train.images = data_train.images(train_ids,:);
data_train.labels = data_train.labels(train_ids);
if (isfield(opts,'fix_unlabeled') && opts.fix_unlabeled)
    unlabeled_ids = train_ids;
else
    % unlabeled images are independent
    unlabeled_ids = 1:size(data_train.unlabeled_images,1);
    unlabeled_ids = unlabeled_ids(randperm(length(unlabeled_ids)));
end
data_train.unlabeled_images = data_train.unlabeled_images(unlabeled_ids,:);
data_train.unlabeled_images_whitened = data_train.unlabeled_images_whitened(unlabeled_ids,:);

fprintf('dataset %s: \n', upper('statistics'))
fprintf('training id min = %d, max = %d \n', min(train_ids), max(train_ids))
print_data_stats(data_train, data_test);

if (strcmpi(opts.dataset,'stl10') && opts.fold_id > 1)
%% STL-10 code for folds 2-10
    fprintf('\n-> processing %s samples \n', upper('training'))
    train_features = forward_pass(data_train.images, net);
    if (net.layers{1}.augment)
        fprintf('\n-> processing %s samples \n', upper('training (augmented)'))
        net.layers{1}.flip = true;
        train_features = cat(1,train_features,forward_pass(data_train.images, net));
        train_labels = repmat(data_train.labels,2,1);
    else
        train_labels = data_train.labels;
    end
    fprintf('\n-> %s with %s \n', upper('classification'), upper(opts.classifier));
    [test_results.acc, test_results.scores, test_results.predicted_labels] = ...
        classifier_committee(train_features, opts.test_features, train_labels, data_test.labels, opts);
    test_results = save_data(test_results, net, opts);
    return;
end

%% Learn filters and connections
for layer_id=1:numel(net.layers), net.layers{layer_id}.stats = []; net.layers{layer_id}.PCA_matrix = []; end
train_features = data_train.unlabeled_images; % use non whitened images to learn filters
for layer_id=1:numel(net.layers)
    if (~isfield(net.layers{layer_id},'connections') || isempty(net.layers{layer_id}.connections))
        fprintf('\nlearning %s from layer %d to layer %d \n', upper('connections'), layer_id-1, layer_id)
        net.layers{layer_id}.connections = learn_connections_unsup(train_features, net.layers{layer_id});
    else
        warning('learning connections from layer %d to layer %d is skipped ', layer_id-1, layer_id)
    end
    if (~isfield(net.layers{layer_id},'filters') || isempty(net.layers{layer_id}.filters))
        fprintf('learning %s for layer %d\n', upper('filters'), layer_id)
        net.layers{layer_id}.filters = learn_filters_unsup(train_features, net.layers{layer_id});
    else
        warning('learning filters for layer %d is skipped ', layer_id)
    end
    if (layer_id < numel(net.layers))
        fprintf('obtaining %s of layer %d \n', upper('feature maps'), layer_id)
        if (layer_id == 1)
            train_features = data_train.unlabeled_images_whitened; % use whitened images to obtain feature maps
        end
        [train_features,stats] = forward_pass(train_features, struct('layers',net.layers{layer_id}));
        if (layer_id < numel(net.layers))
            net.layers{layer_id+1}.sample_size = stats{1}.output_size;
        end
    end
end

fprintf('\nlearning %s and %s for %d layers done \n', upper('connections'), upper('filters'), numel(net.layers))

%% Forward pass for the first N training or unlabeled samples
for layer_id=1:numel(net.layers), net.layers{layer_id}.stats = []; net.layers{layer_id}.PCA_matrix = []; end
fprintf('\n-> processing %s samples \n', upper('training (unlabeled)'))
[train_features, stats] = forward_pass(data_train.unlabeled_images_whitened, net);
opts.PCA_dim(opts.PCA_dim > size(train_features,2)) = [];
    
%% Dimension reduction (PCA) for groups of feature maps
opts.pca_mode = 'pcawhiten';
if (~isfield(opts,'pca_fast') || isempty(opts.pca_fast))
    opts.pca_fast = true;
end
n_max_pca = 7*10^5; % depends on RAM and the number of unlabeled samples
if (size(train_features,2) > n_max_pca)
    fprintf('\n-> %s for groups of feature maps \n', upper('dimension reduction'))
    % perform PCA for the last layer feature map groups independently
    % reshape features to divide them according to the groups
    sz = [net.layers{end}.sample_size(1:2)./net.layers{end}.pool_size, net.layers{end}.n_filters, net.layers{end}.n_groups];
    train_features_reshaped = reshape(train_features(:,1:prod(sz)), [size(train_features,1), sz]);
    % PCA
    opts.pca_dim = min(64,net.layers{end}.n_filters);
    opts.verbose = false;
    n = min(10^4,size(train_features_reshaped,1));
    train_features_split = cell(1,net.layers{end}.n_groups);
    net.layers{end}.PCA_matrix = cell(1,net.layers{end}.n_groups);
    net.layers{end}.data_mean = cell(1,net.layers{end}.n_groups);
    net.layers{end}.L_regul = cell(1,net.layers{end}.n_groups);
    for group = 1:net.layers{end}.n_groups
        [train_features_split{group}, net.layers{end}.PCA_matrix{group}, ...
            net.layers{end}.data_mean{group}, net.layers{end}.L_regul{group}] = ...
            pca_zca_whiten(reshape(train_features_reshaped(1:n,:,:,:,group), [n, prod(sz(1:2))*net.layers{end}.n_filters]), opts);
        fprintf('pca %d->%d for group = %d \n', size(net.layers{end}.PCA_matrix{group}), group)
    end
    
    % normalize features, concatenate with lower layer features
    train_features_reshaped = feature_scaling(cat(2,train_features_split{:}), opts.norm);
    train_features = feature_scaling(cat(2,train_features_reshaped,feature_scaling(train_features(:,prod(sz)+1:end),opts.norm)),opts.norm);
    clear train_features_split
    clear train_features_reshaped
elseif (size(train_features,1) > 10^3)
    opts.pca_dim = max(opts.PCA_dim);
    opts.verbose = true;
    [train_features, net.layers{end}.PCA_matrix, net.layers{end}.data_mean, net.layers{end}.L_regul] = ...
        pca_zca_whiten(train_features, opts);
end

%% Forward pass for other training and test samples
% copy statistics to process other samples
for layer_id=1:numel(stats), net.layers{layer_id}.stats = stats{layer_id}; end

n = min(size(data_train.unlabeled_images_whitened,1),size(data_train.images,1));
if (norm(data_train.unlabeled_images_whitened(1:n,:) - data_train.images(1:n,:)) > 1e-10)
    fprintf('\n-> processing %s samples \n', upper('training'))
    train_features = forward_pass(data_train.images, net);
elseif (size(data_train.images,1) > size(data_train.unlabeled_images_whitened,1))
    fprintf('\n-> processing the rest of %s samples \n', upper('training'))
    train_features = cat(1,train_features,forward_pass(data_train.images(n+1:end,:), net));
end

fprintf('\n-> processing %s samples \n', upper('test'))
test_features = forward_pass(data_test.images, net);

if (net.layers{1}.augment)
    fprintf('\n-> processing %s samples \n', upper('training (augmented)'))
    net.layers{1}.flip = true;
    train_features = cat(1,train_features,forward_pass(data_train.images, net));
    train_labels = repmat(data_train.labels,2,1);
    if (~isfield(opts,'test_features') || isempty(opts.test_features))
        fprintf('\n-> processing %s samples \n', upper('test (augmented)'))
        test_features = cat(1,test_features,forward_pass(data_test.images, net));
    end
    net.layers{1}.flip = false;
else
    train_labels = data_train.labels;
end

%% Dimension reduction (PCA)
if (size(train_features,2) > max(opts.PCA_dim))
    fprintf('\n-> %s \n', upper('dimension reduction'))
    opts.pca_dim = min(size(train_features,2),max(opts.PCA_dim));
    opts.verbose = true;
    [~, PCA_matrix, data_mean, L_regul] = pca_zca_whiten(train_features(1:min(10^4,size(train_features,1)),:), opts);
    train_features = pca_zca_whiten(train_features, opts, PCA_matrix, data_mean, L_regul);
    test_features = pca_zca_whiten(test_features, opts, PCA_matrix, data_mean, L_regul);
end

%% Classification
fprintf('\n-> %s with %s \n', upper('classification'), upper(opts.classifier));
[test_results.acc, test_results.scores, test_results.predicted_labels] = ...
    classifier_committee(train_features, test_features, train_labels, data_test.labels, opts);
test_results.net = net;

%% Save data
test_results = save_data(test_results, net, opts, test_features);

end

function test_results = save_data(test_results, net, opts, test_features)
folds_str = '';
if (opts.n_folds > 1)
    folds_str = sprintf('_%dfolds', opts.n_folds);
end
test_file_name = fullfile(opts.test_path,sprintf('%s_%d%s_%s.mat', opts.dataset, opts.n_train, folds_str, net.arch))
if (opts.n_folds == 1 && exist(test_file_name','file'))
    warning('file already exists')
    return; % do not overwrite data
end
try
    % prevent saving huge PCA matrices
    for layer_id=1:numel(net.layers) 
        PCA_matrix{layer_id} = net.layers{layer_id}.PCA_matrix; net.layers{layer_id}.PCA_matrix = []; 
        if (isfield(net.layers{layer_id},'data_mean')), data_mean{layer_id} = net.layers{layer_id}.data_mean; net.layers{layer_id}.data_mean = []; end
        if (isfield(net.layers{layer_id},'L_regul')), L_regul{layer_id} = net.layers{layer_id}.L_regul; net.layers{layer_id}.L_regul = []; end
    end
    if (opts.n_folds > 1)
        if (opts.fold_id > 1)
            test = load(test_file_name);
            test.acc{end+1} = test_results.acc;
            test.scores{end+1} = test_results.scores;
            test.predicted_labels{end+1} = test_results.predicted_labels;
            test.net{end+1} = net;
            test_results = test;
        else
            test_results.acc = {test_results.acc};
            test_results.scores = {test_results.scores};
            test_results.predicted_labels = {test_results.predicted_labels};
            test_results.net = {net};
        end
    else
        test_results.net = net;
    end
    save(test_file_name,'-struct','test_results','-v7.3')
catch e 
    warning('error while saving test file: %s', e.message)
end
if (strcmpi(opts.dataset,'stl10') && opts.fold_id == 1)
    test_results.test_features = test_features;
    for layer_id=1:numel(net.layers) 
        net.layers{layer_id}.PCA_matrix = PCA_matrix{layer_id}; 
        net.layers{layer_id}.data_mean = data_mean{layer_id};
        net.layers{layer_id}.L_regul = L_regul{layer_id}; 
    end
    test_results.net = {net};
end

% print (intermediate) results
try
    acc = cat(3,test_results.acc{:});
    acc = [[mean(acc(1,:,:),3);std(acc(1,:,:),0,3)]',[mean(acc(2,:,:),3);std(acc(2,:,:),0,3)]',opts.PCA_dim'./1000]
catch
end

end

function [net, opts] = set_up(net, opts)
try
    D = gpuDevice(1);
    g = gpuArray(rand(100,100));
    fprintf('GPU is OK \n')
    if (~isfield(opts,'gpu')), for k=1:numel(net.layers), net.layers{k}.gpu = true; end; end
catch e
    warning('GPU not available: %s', e.message)
    for k=1:numel(net.layers), net.layers{k}.gpu = false; end
end
if (isfield(opts,'matconvnet') && exist(opts.matconvnet,'dir'))
    addpath(fullfile(opts.matconvnet,'matlab/mex'))
    run(fullfile(opts.matconvnet,'matlab/vl_setupnn.m'))
    vl_nnconv(rand(32,32,3,10,'single'),rand(5,5,3,20,'single'),[]);
    vl_nnconv(gpuArray(rand(32,32,3,10,'single')),gpuArray(rand(5,5,3,20,'single')),[]);
    fprintf('MatConvNet is OK \n')
else
    warning('MatConvNet not found, Matlab implementation will be used')
    for k=1:numel(net.layers), net.layers{k}.is_vl = false; end
end
if (isfield(opts,'vlfeat') && exist(opts.vlfeat,'dir'))
    addpath(opts.vlfeat)
    vl_kmeans(rand(100,5), 2); % check that it works
    fprintf('VLFeat is OK \n')
else
    warning('VLFeat not found, Matlab kmeans implementation will be used to learn filters')
    for k=1:numel(net.layers), net.layers{k}.learning_method = 'kmeans_matlab'; end
end
if (isfield(opts,'gtsvm') && exist(opts.gtsvm,'dir'))
    addpath(opts.gtsvm)
    % check that it works
    context = gtsvm;
    context.initialize( rand(1000,100), randi([0 4],1000,1), true, 1, 'gaussian', 0.05, 0, 0, false );
    context.optimize( 0.01, 1000000 );
    classifications = context.classify( rand(1000,100) );
    opts.classifier = 'gtsvm';
    fprintf('GTSVM is OK \n')
elseif (isfield(opts,'libsvm') && exist(opts.libsvm,'dir'))
    addpath(opts.libsvm)
    svmtrain(randi(5,100,1),rand(100,100),'-q'); % check that it works
    opts.classifier = 'libsvm';
    fprintf('LIBSVM is OK \n')
else
    opts.classifier = 'lda';
    warning('LIBSVM or GTSVM should be installed, Matlab LDA implementation will be used for classification')
end

if (~isfield(opts,'test_path'))
    opts.test_path = fullfile(opts.dataDir,'test_results');
end
if (~exist(opts.test_path,'dir'))
    mkdir(opts.test_path)
end
addpath(opts.test_path)

end

function print_data_stats(data_train, data_test)
fprintf('training labels: %s \n', num2str(unique(data_train.labels)')) 
fprintf('test labels: %s \n', num2str(unique(data_test.labels)')) 
if (any(unique(data_train.labels) ~= unique(data_test.labels)))
    error('invalid labels')
end
for label = min(data_train.labels):max(data_train.labels)
    fprintf('label %d, N training: %d, N test: %d \n', label, nnz(data_train.labels == label), nnz(data_test.labels == label));
end
fprintf('total N training: %d, N test %d \n', length(data_train.labels), length(data_test.labels))

fprintf('checksum for training sample 1: %5.3f \n', norm(data_train.images(1,:)));
fprintf('checksum for the last training sample: %5.3f \n', norm(data_train.images(end,:)));
fprintf('checksum for test sample 1: %5.3f \n', norm(data_test.images(1,:)));
fprintf('checksum for the last test sample: %5.3f \n', norm(data_test.images(end,:)));
mn_sd1 = [mean(data_train.images(:)),std(data_train.images(:))];
fprintf('train samples mean and std: %3.3f, %3.3f \n', mn_sd1)
mn_sd2 = [mean(data_test.images(:)),std(data_test.images(:))];
fprintf('test samples mean and std: %3.3f, %3.3f \n', mn_sd2)
mn_sd3 = [mean(data_train.unlabeled_images(:)), std(data_train.unlabeled_images(:))];
fprintf('unlabeled samples mean and std: %3.3f, %3.3f \n', mn_sd3)
mn_sd4 = [mean(data_train.unlabeled_images_whitened(:)),std(data_train.unlabeled_images_whitened(:))];
fprintf('unlabeled samples (whitened) mean and std: %3.3f, %3.3f \n', mn_sd4)

n = min([1000, size(data_train.images,1), size(data_test.images,1)]);
D = pdist2(data_train.images(randperm(size(data_train.images,1),n),:),...
    data_test.images(randperm(size(data_test.images,1),n),:));
m = min(D(:));
fprintf('min distance between 1k random training and test samples: %3.3f \n', m)
if (m < 1e-5)
    error('training and test samples might overlap')
end
n = min([1000, size(data_train.unlabeled_images,1), size(data_test.images,1)]);
D = pdist2(data_train.unlabeled_images(randperm(size(data_train.unlabeled_images,1),n),:),...
    data_test.images(randperm(size(data_test.images,1),n),:));
m = min(D(:));
fprintf('min distance between 1k random unlabeled and test samples: %3.3f \n', m)
if (m < 1e-5)
    warning('unlabeled and test samples might overlap')
end
n = min([1000, size(data_train.unlabeled_images_whitened,1), size(data_test.images,1)]);
D = pdist2(data_train.unlabeled_images_whitened(randperm(size(data_train.unlabeled_images_whitened,1),n),:),...
    data_test.images(randperm(size(data_test.images,1),n),:));
m = min(D(:));
fprintf('min distance between 1k random unlabeled (whitened) and test samples: %3.3f \n', m)
if (m < 1e-5)
    warning('unlabeled (whitened) and test samples might overlap')
end
end
