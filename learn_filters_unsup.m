function [filters, params] = learn_filters_unsup(feature_maps, opts)
% Learns filters for some layer l in the unsupervised way
% 
% feature_maps - feature maps of layer l-1 (input samples to layer l), rows are samples, columns are variables, 
% there are prod(opts.sample_size) columns
% opts - a structure with learning method parameters (can be empty to use default values in case of CIFAR-10)
% opts.connections - a binary connection matrix with n_groups rows and N_filters_{l-1} columns
% filters - an array of cells (one cell per group), each cell contains a 4D array (height x width x depth x opts.n_filters)
% filter depth (number of channels) is defined by opts.connections (if not specified, equals opts.sample_size(end))
% params - an array of cells (one cell per group) with vectors of the joint spatial and temporal resolutions for each filter
%
% Currently, the only supported methods are kmeans, kmedoids and GMM
% It's highly recommended to install VlFeat (http://www.vlfeat.org/) beforehand, 
% because it tends to be much faster than Matlab implementations

%% Set data dependent parameters
% the following set of parameters should be specified according to your data
% the default values are suitable only for datasets like CIFAR-10 and a single layer architecture
fprintf('-> setting default parameters... \n')

if (~isfield(opts,'sample_size'))
    opts.sample_size = [32,32,3];
    fprintf('sample_size: \t\t %dx%dx%d \n', opts.sample_size)
end
if (length(opts.sample_size) > 3)
    opts.sample_size(opts.sample_size == 1) = [];
elseif (length(opts.sample_size) < 3)
    opts.sample_size = [opts.sample_size,1];
end

if (~isfield(opts,'n_filters'))
    opts.n_filters = 256;
    fprintf('n_filters: \t\t %d \n', opts.n_filters)
end
if (~isfield(opts,'n_groups'))
    opts.n_groups = 1; % the number of feature map groups, n_groups = 1 corresponds to a typical CNN architecture 
    fprintf('n_groups: \t\t %d \n', opts.n_groups)
end
if (~isfield(opts,'filter_size'))
    opts.filter_size = [13,13,3];
    fprintf('filter_size: \t\t %dx%dx%d \n', opts.filter_size)
end
if (~isfield(opts,'crop_size'))
    opts.crop_size = opts.filter_size(1);
    fprintf('crop_size: \t\t %d \n', opts.crop_size)
end
if (~isfield(opts,'conv_orders'))
    opts.conv_orders = 0:4; % autoconvolution orders (n=0,1,2,3,4)
    fprintf('conv_orders: \t\t %s \n', num2str(opts.conv_orders))
end
if (~isfield(opts,'norm_type'))
    opts.norm_type = 1;   % 1 should be to train the first layer filters, 2 - for higher layers
    fprintf('norm_type: \t\t %d \n', opts.norm_type)
end

if (~isfield(opts,'connections'))
    connections = true(1,opts.sample_size(end));
else
    connections = opts.connections;
end
n_groups = size(connections,1);

%% Set default values of data independent parameters
if (~isfield(opts,'patches_norm'))
    opts.patches_norm = 'gray'; % normalize patches in the range [0,1] before (whitening and) learning 
    fprintf('patches_norm: \t\t ''%s'' \n', opts.patches_norm)
end

if (~isfield(opts,'whiten'))
    opts.whiten = true; % true to whiten patches before learning
    fprintf('whiten: \t\t %d \n', opts.whiten)
end
if (~isfield(opts,'whiten_independ'))
    opts.whiten_independ = false; % true to whiten patches before learning independently for each autoconvolution order
    fprintf('whiten_independ: \t %d \n', opts.whiten_independ)
end
opts.pca_epsilon = 0.05; % whitening regularization constant
opts.pca_mode = 'zcawhiten'; % for patches whitening
    
if (~isfield(opts,'shared_filters'))
    opts.shared_filters = true; % true to use same filters for all groups of feature maps
    fprintf('shared_filters: \t %d \n', opts.shared_filters)
end

if (~isfield(opts,'learning_method'))
    opts.learning_method = 'kmeans'; % clustering with k-means
    fprintf('learning_method: \t ''%s'' \n', opts.learning_method)
end

if (strcmpi(opts.learning_method,'kmeans'))
    if (~isfield(opts,'kmeans_algorithm'))
        opts.kmeans_algorithm = 'ELKAN'; % 'ANN' or 'ELKAN'
        fprintf('kmeans_algorithm: \t ''%s'' \n', opts.kmeans_algorithm)
    end
end

if (~isfield(opts,'batch_size'))
    opts.batch_size = 256; % extract batches of patches (to speed up)
    fprintf('batch_size: \t\t %d \n', opts.batch_size)
end

if (~isfield(opts,'gpu'))
    opts.gpu = 0; % setting to true might help to extract large patches faster
    fprintf('gpu: \t\t\t %d \n', opts.gpu)
end

if (~isfield(opts,'vis'))
    opts.vis = false; % visualization of filters and connections
    fprintf('vis: \t\t\t %d \n', opts.vis)
end

%% Extract autoconvolutional patches
n_min = 0.5*10^5;
fprintf('-> extracting at least %d patches for %d group(s)... \n', n_min, n_groups)
nSamples = size(feature_maps,1);
patches = cell(length(opts.conv_orders),n_groups);
n_min_group = ceil(n_min/max(1,opts.shared_filters*n_groups));
opts.batch_size = min([opts.batch_size, nSamples, n_min_group]);
for group=1:n_groups
    fprintf('group: %d/%d, feature maps: %s \n', group, n_groups, num2str(find(connections(group,:))));
    n_patches = 0;
    offset = 0;
    patches_batch = zeros(opts.crop_size(1),opts.crop_size(1),nnz(connections(group,:)),opts.batch_size,'single');
    if (opts.gpu)
        patches_batch = gpuArray(patches_batch);
    end
    while (n_patches < n_min_group)
        % featMaps - a 4D array (spatial rows x cols x N_filters_prev x batch_size)
        featMaps = reshape(feature_maps(randperm(nSamples, opts.batch_size),:)', [opts.sample_size, opts.batch_size]);
        if (opts.gpu)
            featMaps = gpuArray(featMaps);
        end
        % take crops with random spatial locations
        rows = randi([1+offset, 1+opts.sample_size(1)-opts.crop_size(1)-offset], 1, opts.batch_size);
        cols = randi([1+offset, 1+opts.sample_size(2)-opts.crop_size(1)-offset], 1, opts.batch_size);
        for b=1:opts.batch_size
            patches_batch(:,:,:,b) = featMaps(rows(b):rows(b)+opts.crop_size(1)-1, ...
                cols(b):cols(b)+opts.crop_size(1)-1, connections(group,:), b);
        end
        % X_n - a cell with 4D arrays (spatial rows x cols x depth x batch_size)
        X_n = autoconv_recursive_2d(patches_batch, max(opts.conv_orders), opts.filter_size, opts.norm_type);
        X_n = X_n(opts.conv_orders+1); % take patches of only specified orders
        
        % collect patches independently for each conv_order (n)
        for n=1:numel(X_n)
            if (opts.gpu)
                X_n{n} = gather(X_n{n}); 
            end
            % estimate parameters to find and remove invalid patches
            params_sample = estimate_params(X_n{n});
            X_n{n}(:,:,:,cellfun(@isempty,params_sample)) = [];
            % concatenate into the global set
            patches{n,group} = cat(4,patches{n,group},X_n{n});
            n_patches = n_patches + size(X_n{n},4);
        end
    end
end
clear feature_maps;

%% Learn filters
if (opts.shared_filters), n_groups = 1; else n_groups = opts.n_groups; end
params = cell(n_groups,1);
filters = cell(n_groups,1);
if (opts.shared_filters)
    for n=1:size(patches,1)
        patches{n,1} = cat(4,patches{n,:});
    end
    patches = patches(:,1);
end
if (opts.whiten_independ)
    pca_fractions = [0.90,0.95,0.97,0.98,0.99]; % for n=0, 0.90 is good, but for n > 0, 0.95 and larger is better
else
    pca_fractions = 0.99;
    for group = 1:n_groups
        patches{1,group} = cat(4,patches{:,group});
    end
    patches = patches(1,:);
end
% patches normalization and whitening
for group = 1:size(patches,2)
    for n=1:size(patches,1)
        sz = size(patches{n,group});
        fprintf('%d patches are extracted \n', sz(4));
        patches{n,group} = reshape(real(patches{n,group}),[prod(sz(1:3)),sz(4)])';
        if (~isempty(opts.patches_norm))
            % important before whitening
            fprintf('-> %s-normalization of patches \n', opts.patches_norm);
            patches{n,group} = feature_scaling(patches{n,group}, opts.patches_norm);
        end
        if (opts.whiten)
            if (opts.whiten_independ)
                opts.pca_fraction = pca_fractions(opts.conv_orders(n)+1); 
            else
                opts.pca_fraction = pca_fractions(1);
            end
            patches{n,group} = pca_zca_whiten(patches{n,group}, opts);
        end
    end
    patches_all = cat(1,patches{:,group});
    if (opts.whiten_independ)
        % important before whitening
        fprintf('-> %s-normalization of patches \n', opts.patches_norm);
        patches_all = feature_scaling(patches_all, opts.patches_norm);
        opts.pca_fraction = 0.99;
        patches_all = pca_zca_whiten(patches_all, opts);
    end
    
    % learn filters (k-means, k-medoids, ICA, etc.) for the current group
%     patches_all = feature_scaling(patches_all, 'l2'); % normalization before clustering leads
    % to more uniform clusters and better looking filters, but surprisingly, hurts classification accuracy
    filters_clusters = learn_filters(patches_all, opts);
    filters_clusters = feature_scaling(filters_clusters, 'l2'); % normalize for better usage as convolution kernels
    
    % estimate the joint spatial and frequency resolution
    filters{group} = permute(reshape(filters_clusters,[size(filters_clusters,1),sz(1:3)]), [2:4,1]);
    params{group} = cell2mat(estimate_params(filters{group}));
    % sort by the joint spatial and frequency resolution
    [params{group},ids] = sort(params{group},'ascend');
    filters{group} = filters{group}(:,:,:,ids);
end
fprintf('filters are learned for %d group(s) \n', opts.n_groups)

end

function X_n = autoconv_recursive_2d(X, conv_order_MAX, filter_size, norm_type)
% X - a random image (patch) or batches in the spatial domain
% n_MAX - the last autoconvolution order
% filter_size - desired size of returned patches
% X_n - a collection of autoconvolutional patches of orders n=0,1,...,n_MAX
% Patches are normalized in the range [0,1]

samplingRatio = 2;
X_n = cell(1,conv_order_MAX+1);
y_min = min(min(min(X,[],1),[],2),[],3);
y_max = max(max(max(X,[],1),[],2),[],3);
for conv_order = 0:conv_order_MAX
    if (conv_order > 0)
        X = real(autoconv_2d(X, norm_type));
        if (conv_order > 1)
            if (rand > 0.5 || conv_order == 3)
                X = imresize(X,1/samplingRatio); 
            else
                X = downsample(X, samplingRatio, 'space');
            end
        end
        X = change_range(real(X), y_min, y_max, norm_type);
        X_n{1,conv_order+1} = single(change_range(imresize(X,filter_size(1)/size(X,1)), y_min, y_max, norm_type));
    else
        if (size(X,1) ~= filter_size(1))
            y_tmp = imresize(X, filter_size(1)/size(X,1));
            y_tmp = change_range(y_tmp, y_min, y_max, norm_type);
            if (conv_order_MAX == 0)
                X = y_tmp;
            end
        else
            y_tmp = X;
        end
        X_n{1,conv_order+1} = y_tmp;
    end
end
end

function X = autoconv_2d(X, norm_type)
% input X - an input image in the spatial domain
% output X - a result in the spatial domain of convolving X with itself
% X can be a batch of images, the first two dimensions must be the spatial (rows,columns) ones

if (norm_type == 1)
    m = mean(mean(X,1),2);
    X = bsxfun(@minus,X,m);
    sd = std(std(X,0,1),0,2);
    X = bsxfun(@rdivide,X,sd+1e-5);
else
    m = mean(mean(mean(X,1),2),3);
    X = bsxfun(@minus,X,m);
end
sz = size(X);
X = padarray(X, sz(1:2)-1, 'post'); % zero-padding to compute linear convolution
X = ifft2(fft2(X).^2); % autoconvolution in the frequency domain 
end

function y = change_range(y, y_min, y_max, norm_type)
if (norm_type ~= 1)
    m1 = min(min(min(y,[],1),[],2),[],3);
    m2 = max(max(max(y,[],1),[],2),[],3);
    y = bsxfun(@plus,bsxfun(@times,bsxfun(@rdivide,bsxfun(@minus,y,m1),bsxfun(@minus,m2,m1)),bsxfun(@minus,y_max,y_min)),y_min);
end
end

function [filters, cluster_ids, clusters_weights] = learn_filters(data, opts)

sz = size(data);
if (sz(1) < opts.n_filters)
    error('too few data points')
end
fprintf('-> learning %d filters from data points of size %dx%d using ''%s''... \n', opts.n_filters, sz(1:2), opts.learning_method)
     
cluster_ids = [];
clusters_weights = [];

if (strcmpi(opts.learning_method,'random'))
    filters = data(randperm(sz(1),opts.n_filters),:);
elseif (strcmpi(opts.learning_method,'vl_gmm'))
    [filters,~,~] = vl_gmm(data', opts.n_filters);
    filters = filters';
elseif (strcmpi(opts.learning_method,'kmedoids'))
    % this can be very slow, however, we can enjoy various distance measures
    opts_k = statset('kmedoids');
    opts_k.UseParallel = true;
    opts_k.MaxIter = 200;
    [ids,~,~,~,cluster_ids] = kmedoids(data, opts.n_filters, 'Options', opts_k, 'Distance', 'cosine', 'Replicates',4);
    filters = data(cluster_ids,:);
elseif (~isempty(strfind(opts.learning_method,'kmeans')))
    if (strcmpi(opts.learning_method,'kmeans'))
        [filters,ids,energy] = vl_kmeans(data', opts.n_filters, 'Algorithm', opts.kmeans_algorithm,'Distance','l2',...
                'NumRepetitions',3,'MaxNumComparisons',2000,'MaxNumIterations',1000,'Initialization','PLUSPLUS');
        filters = filters';
    elseif (strcmpi(opts.learning_method,'kmeans_matlab'))
        % this can be very slow, however, we can enjoy various distance measures
        opts_k = statset('kmeans');
        opts_k.UseParallel = true;
        [ids,filters,sumd,D]  = kmeans(data, opts.n_filters,'Distance','cosine','Replicates',4,'MaxIter',200,'Display','final','Options',opts_k);
        delete(gcp)
    else
        error('not supported learning method')
    end
    
    % collect some statistics of clusters
    cluster_ids = zeros(min(sz(1),100), opts.n_filters); % the closest data points to clusters
    clusters_weights = zeros(1,opts.n_filters,'uint32'); % the number of data points in each cluster
    for clust=1:opts.n_filters
        ro = sum(bsxfun(@minus,data,filters(clust,:)).^2,2);
        [~,id] = sort(ro);
        cluster_ids(:,clust) = id(1:min(length(id),100));
        clusters_weights(clust) = nnz(ids == clust);
    end
    fprintf('number of clusters with the min (%d) and max (%d) number of data points in clusters: %d and %d \n', ...
        min(clusters_weights), max(clusters_weights), nnz(clusters_weights == min(clusters_weights)), ...
        nnz(clusters_weights == max(clusters_weights)))
else
    error('not supported learning method')
end

end

function params = estimate_params(filters)
n_filters = size(filters,4);
params = cell(n_filters,1);
sz = size(filters);
T = 1;
[axes{1},axes{2}] = meshgrid(1/sz(1)*(-floor(sz(1)/2):T:ceil(sz(1)/2)-1), 1/sz(2)*(-floor(sz(2)/2):T:ceil(sz(2)/2)-1)); % frequency axes
[axes{3},axes{4}] = meshgrid(-sz(1)/2:T:sz(1)/2-1, -sz(2)/2:T:sz(2)/2-1); % spatial axes
for k=1:numel(axes)
    axes{k} = repmat(axes{k},1,1,sz(3));
    axes{k} = axes{k}(:);
end
axes_freq = cat(2,axes{1:2});
axes_spatial = cat(2,axes{3:4});
for i=1:n_filters
    filters(:,:,:,i) = reshape(hilbert(reshape(filters(:,:,:,i),[prod(sz(1:3)),1])),size(filters(:,:,:,i)));
end
F = fftshift(fftshift(fft2(filters),1),2);
F_abs_org = reshape(F.*conj(F),[prod(sz(1:3)),sz(4)]);
filters_abs_org = reshape(filters.*conj(filters),[prod(sz(1:3)),sz(4)]);
    
for i=1:n_filters
    try
        spatial = effective_width(filters_abs_org(:,i), axes_spatial);
        freq = effective_width(F_abs_org(:,i), axes_freq);
        sp = spatial.width.*freq.width;
        params{i} = sp; % other parameters can be added, so we keep it cell
    catch e
        % e.g., in case variance is zero 
        params{i} = [];
    end
end

end

function widthData  = effective_width(X, axes)
% X and axes must be column vectors of the same length

if (min(size(X)) > 1)
    X = X(:);
end
weights = (X./sum(X)).';
m_w = weights*axes;
axes = bsxfun(@minus,axes,m_w);
weighted_cov = [weights.*axes(:,1)';(weights.*axes(:,2)')]*axes;
[eig_vectors, eig_values] = eig(weighted_cov);
widthData.s = sqrt(diag(eig_values));
widthData.eigs = eig_vectors;
widthData.w_cov = weighted_cov;
widthData.w_means = m_w;
widthData.width = prod(widthData.s);
end

function f = downsample(f, dwn_coef, type, varargin)
% This is a quite general function to take a central part of some signal f with some downsampling coefficient dwn_coef.
% type can be 'freq', otherwise assumed 'spatial'
% varargin can be used to specify the number of dimensions along which downsampling is performed
% the size of output f is defined as size(f)/dwn_coef

if (nargin <= 3)
    n_dimensions = 2;
else
    n_dimensions = varargin{1};
end

if (n_dimensions > 3)
    error('maximum 3 dimensions is supported')
end

if (length(dwn_coef) == 1)
    dwn_coef = repmat(dwn_coef,1,n_dimensions);
elseif (length(dwn_coef) == 2)
    dwn_coef = [dwn_coef,1];
end
if (isequal(lower(type),'freq'))
    f = fftshift(f);
end
sz = size(f);
sz = sz(1:n_dimensions);
sz_new = round(sz./dwn_coef(1:n_dimensions));
d = repmat((sz-sz_new)./2,2,1);
for i=1:n_dimensions
    if (abs(d(1,i)-floor(d(1,i))) > eps)
        d(1,i) = ceil(d(1,i));
        d(2,i) = floor(d(2,i));
    end
end
f = f(d(1,1)+1:end-d(2,1), d(1,2)+1:end-d(2,2), :, :, :);
if (n_dimensions >= 3)
    f = f(:,:,d(1,3)+1:end-d(2,3),:,:);
end
if (isequal(lower(type),'freq'))
    f = ifftshift(f);
end
end