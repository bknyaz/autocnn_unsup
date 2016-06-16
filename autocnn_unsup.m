% AutoCNN main pipeline suitable for image datasets such as MNIST, CIFAR, STL-10

time_start = clock;
fprintf('\ntest %s on %s \n', upper('started'), datestr(time_start))

%% Learn filters and connections
train_features = data_train.unlabeled_images; % use non whitened images to learn filters
for layer_id=1:numel(net.layers)
    fprintf('\nlearning %s from layer %d to layer %d \n', upper('connections'), layer_id-1, layer_id)
    net.layers{layer_id}.connections = learn_connections_unsup(train_features, net.layers{layer_id});
    fprintf('learning %s for layer %d\n', upper('filters'), layer_id)
    net.layers{layer_id}.filters = learn_filters_unsup(train_features, net.layers{layer_id});
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
fprintf('\n-> processing %s samples \n', upper('training'))
[train_features, stats] = forward_pass(data_train.unlabeled_images_whitened, net);

%% Dimension reduction (PCA) for groups of feature maps
opts.norm = 'stat';
opts.pca_mode = 'pcawhiten';
opts.PCA_dim = PCA_dim; % PCA dimensionalities (p_j) for the SVM committee
opts.pca_fast = true;
n_max_pca = 350*10^3;
if (size(train_features,2) > n_max_pca)
    fprintf('\n-> %s for groups of feature maps \n', upper('dimension reduction'))
    % perform PCA for the last layer feature map groups independently
    % reshape features to divide them according to the groups
    sz = [net.layers{end}.sample_size(1:2)./net.layers{end}.pool_size, net.layers{end}.n_filters, net.layers{end}.n_groups];
    train_features_reshaped = reshape(train_features(:,1:prod(sz)), [size(train_features,1), sz]);
    % PCA
    opts.pca_dim = net.layers{end}.n_filters;
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
        fprintf('group = %d \n', group)
    end
    
    % normalize features, concatenate with lower layer features
    train_features_reshaped = feature_scaling(cat(2,train_features_split{:}), opts.norm);
    train_features = feature_scaling(cat(2,train_features_reshaped,feature_scaling(train_features(:,prod(sz)+1:end),opts.norm)),opts.norm);
    clear train_features_split
    clear train_features_reshaped
elseif (size(data_train.images,1) > 10^4)
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
    [train_features, stats] = forward_pass(data_train.images, net);
elseif (size(data_train.images,1) > size(data_train.unlabeled_images_whitened,1))
    train_features = cat(1,train_features,forward_pass(data_train.images(n+1:end,:), net));
end
fprintf('\n-> processing %s samples \n', upper('test'))
test_features = forward_pass(data_test.images, net);

if (net.layers{1}.augment)
    fprintf('\n-> processing %s samples \n', upper('training (augmented)'))
    net.layers{1}.flip = true;
    train_features = cat(1,train_features,forward_pass(data_train.images, net));
    train_labels = repmat(train_labels,2,1);
else
    train_labels = data_train.labels;
end

%% Dimension reduction (PCA)
if (size(train_features,2) > max(opts.PCA_dim))
    fprintf('\n-> %s \n', upper('dimension reduction'))
    opts.pca_dim = max(opts.PCA_dim);
    opts.verbose = true;
    [~, PCA_matrix, data_mean, L_regul] = pca_zca_whiten(train_features(1:min(10^4,size(train_features,1)),:), opts);
    train_features = pca_zca_whiten(train_features, opts, PCA_matrix, data_mean, L_regul);
    test_features = pca_zca_whiten(test_features, opts, PCA_matrix, data_mean, L_regul);
end

%% Classification
fprintf('\n-> %s with SVMs \n', upper('classification'))
opts.gpu_svm = true;
[acc, scores, predicted_labels] = SVM_committee(train_features, test_features, train_labels, data_test.labels, opts);

fprintf('test took %5.3f seconds \n', etime(clock,time_start));
fprintf('test %s on %s \n\n', upper('finished'), datestr(clock))