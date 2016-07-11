function [fmaps_out, stats] = forward_pass(feature_maps, net)
% Processes samples (images) 'feature_maps' according to the multilayer architecture of network 'net'
% Uses Matconvnet for faster convolutions and other standard operations (e.g., pooling)
% 
% feature_maps - a matrix, in which rows are samples and columns are variables (features)
% stats - statistics obtained from the training samples later used for the test samples
% net.layers - a cell array of layer parameters
% Each layer contains filters (a 5d array) and connections (a 2d array) and stats (in case of test samples),
% other model parameters are fixed (not learned)

% init variables
time_global = tic;
if (~iscell(net.layers))
    net.layers = {net.layers};
end
n_layers = numel(net.layers);
stats = cell(1,n_layers);
% feature_maps_multi = cell(1,n_layers);

n_samples = size(feature_maps,1);
n_batches = ceil(n_samples/net.layers{1}.batch_size);

features = {feature_maps(1:min(n_samples,2),:)};
if (net.layers{1}.gpu)
    features{1} = gpuArray(features{1});
end
stats_size = cell(1,n_layers);
% process 2 samples layer wise to check variables and preallocate arrays
feature_length = zeros(1,n_layers);
fmaps_out_multi = cell(1,n_layers);
fprintf('checking and preparing parameters and data \n')
for layer_id = 1:n_layers
    if (iscell(net.layers{layer_id}.filters))
        net.layers{layer_id}.filters = single(cat(5,net.layers{layer_id}.filters{:}));
    end
    sz_filters = size(net.layers{layer_id}.filters);
    if (length(sz_filters) < 5), sz_filters(5) = 1; end;
    fprintf('-> %s %d: %d feature maps from layer %d used, %d groups, filters %dx%dx%dx%dx%d \n', ...
        upper('layer'), layer_id, nnz(sum(net.layers{layer_id}.connections,1) > 1e-10), layer_id-1, ...
        size(net.layers{layer_id}.connections,1), sz_filters)
    
    net.layers{layer_id} = set_default_values(net.layers{layer_id});
    
    % for the last layer multidictionary features are not applicable
    if (layer_id == n_layers), net.layers{layer_id}.multidict = false; end % override the value
    
    if (~isfield(net.layers{layer_id},'connections_next') && layer_id < n_layers && net.layers{layer_id}.pruned)
        net.layers{layer_id}.connections_next = net.layers{layer_id+1}.connections; % to prune features
    else
        net.layers{layer_id}.connections_next = [];
    end
    
    if (~isfield(net.layers{layer_id},'lcn'))
        % for the last layer local contrast normalization is not useful
        net.layers{layer_id}.lcn = layer_id < n_layers;
    else
        if (layer_id == n_layers), net.layers{layer_id}.lcn = false; end % override the value
    end
    
    if (~isfield(net.layers{layer_id},'conv_pad'))
        net.layers{layer_id}.conv_pad = floor(net.layers{layer_id}.filter_size(1:2)./2); % zero padding for convolution
    end
    if (~isfield(net.layers{layer_id},'pool_pad'))
        m = mod(net.layers{layer_id}.sample_size(1)./net.layers{layer_id}.conv_stride, net.layers{layer_id}.pool_size);
        net.layers{layer_id}.pool_pad = m;
        if (m > 0)
            net.layers{layer_id}.pool_pad = (net.layers{layer_id}.pool_size-m)/2;
        end
    end

    % prepare filters
    if (~net.layers{layer_id}.is_vl)
        net.layers{layer_id}.filters = padarray(net.layers{layer_id}.filters, ...
            net.layers{layer_id}.sample_size(1:2)-1 + net.layers{layer_id}.conv_pad - ...
            floor(net.layers{layer_id}.filter_size(1:2)./2),0,'post');
    end

    if (net.layers{layer_id}.complex_filters)
        net.layers{layer_id}.filters = hilbert_5d(net.layers{layer_id}.filters);
    end
    net.layers{layer_id}.filters = norm_5d(net.layers{layer_id}.filters); % normalize filters [and send them to a GPU]
    if (net.layers{layer_id}.gpu)
        net.layers{layer_id}.filters = gpuArray(net.layers{layer_id}.filters);
    end
    
    if (~net.layers{layer_id}.is_vl)
        net.layers{layer_id}.dims = 1:2;
        for d=net.layers{layer_id}.dims, net.layers{layer_id}.filters = fft(net.layers{layer_id}.filters,[],d); end
    end

    % Reorganize feature maps according to the connection matrix for memory efficiency
    % the connection matrix has net.layers{layer_id}.n_groups rows and the number of columns
    % equals the number of filters in the previous layer, i.e. N_filters_{l-1}
    % filters is a 5d array: rows x cols x depth x N_filters x n_groups, although depth and n_groups can be 1
    % feature_maps is a 2d array, which can be reshaped into a 4d array: n_samples x rows x cols x N_filters_{l-1}
    redundant_features = sum(net.layers{layer_id}.connections,1) < max(net.layers{layer_id}.connections(:))*1e-5;
    net.layers{layer_id}.connections = net.layers{layer_id}.connections(:,~redundant_features);
    net.layers{layer_id}.redundant_features = redundant_features;

    % determine the size of feature_maps_out and preallocate an array accordingly
    % print checksum for the first 2 samples
    features_check = cell(size(features{1},1),1);
    for sample_id=1:size(features{1},1)
        [features_check{sample_id}, stats_size{layer_id}] = forward_pass_batch(features{1}(sample_id,:), net.layers{layer_id}.filters, net.layers{layer_id});
        fprintf('checksum for sample %d = %f \n', sample_id, norm(features_check{sample_id}{1}(:)));
    end
    % get size of feature maps
    if (isempty(net.layers{layer_id}.stats))
        if (length(stats_size{layer_id}{1}.output_size{1}) < 4)
            stats_size{layer_id}{1}.output_size{1} = [stats_size{layer_id}{1}.output_size{1}(1:2),1,stats_size{layer_id}{1}.output_size{1}(3)];
        end
        if (layer_id < n_layers)
            net.layers{layer_id+1}.sample_size = stats_size{layer_id}{1}.output_size{1};
        end
    end
    
    stats{layer_id} = cell(1,n_batches);

    features{1} = cat(1,features_check{1}{1},features_check{2}{1});
    if (~isempty(net.layers{layer_id}.norm))
        for k=1:numel(features), features{k} = feature_scaling(features{k}, net.layers{layer_id}.norm); end
    end
    if (net.layers{layer_id}.multidict)
        fmaps_out_multi{layer_id} = cat(1,features_check{1}{2},features_check{2}{2});
        feature_length(layer_id+1) = size(fmaps_out_multi{layer_id},2);
        fprintf('layer %d feature maps: %dx%d \n', layer_id, n_samples, feature_length(layer_id+1))
    end
    feature_length(1) = size(features{1},2);
end

% Dimension reduction (optional)
features = pca_whiten_wrap(cat(2,features{1},fmaps_out_multi{:}), net.layers{layer_id});
    
fmaps_out = zeros(n_samples, size(features,2), 'single');
fprintf('-> (multidict) feature maps: %dx%d (%s) \n', size(fmaps_out), class(fmaps_out))

fprintf('processing batches \n')    
for batch_id = 1:n_batches
    time = tic; % to measure forward pass speed
    samples_ids = max(1,min(n_samples, (batch_id-1)*net.layers{1}.batch_size+1:batch_id*net.layers{1}.batch_size));
    features = feature_maps(samples_ids,:);
    fmaps_out_batch = zeros(length(samples_ids),sum(feature_length),'single');
    if (net.layers{1}.gpu), features = gpuArray(features); end
    
    for layer_id = 1:n_layers
        
        if (layer_id > 1 && isfield(net.layers{layer_id},'lcn_l2') && net.layers{layer_id}.lcn_l2)
            % useful for MNIST: scale feature maps before passing to the next layer
            features = local_fmaps_norm(features, net.layers{layer_id}.sample_size);
        end
        % process a single batch with an AutoCNN
        [features, stats{layer_id}{batch_id}] = forward_pass_batch(features, net.layers{layer_id}.filters, net.layers{layer_id});
        % apply feature normalization for each sample (except for the last layer)
        if (~isempty(net.layers{layer_id}.norm) && layer_id < n_layers)
            features{1} = feature_scaling(features{1}, net.layers{layer_id}.norm);
        end
        if (net.layers{layer_id}.multidict)
            fmaps_out_batch(:,sum(feature_length(1:layer_id))+1:sum(feature_length(1:layer_id+1))) = gather(features{2});
        end
        features = features{1};
    end
    fmaps_out_batch(:,1:feature_length(1)) = gather(features);
    % normalize concatenated features
    if (~isempty(net.layers{layer_id}.norm))
        fmaps_out_batch = feature_scaling(fmaps_out_batch, net.layers{layer_id}.norm);
    end
    
    % Dimension reduction (optional)
    fmaps_out(samples_ids,:) = pca_whiten_wrap(fmaps_out_batch, net.layers{layer_id});
        
    time = toc(time);
    if (mod(batch_id,net.layers{layer_id}.progress_print)==0), fprintf('batch %d/%d, %3.3f samples/sec \n', batch_id, n_batches, length(samples_ids)/time); end
end

clear feature_maps;

% collect feature maps statistics if requested
if (nargout > 1)
    for layer_id = 1:n_layers
        if (isfield(net.layers{layer_id},'stats') && ~isempty(net.layers{layer_id}.stats))
            continue;
        end
        fprintf('collecting statistics for layer %d \n', layer_id)
        means = []; % mean values of all batches
        stds = []; % std values of all batches
        lcn_means = []; % LCN weighted standard deviations
        % filter responses mean, min and max values
        feat_means = [];
        feat_mins = [];
        feat_maxs = [];
        for batch=1:numel(stats{layer_id}) % collect for batches
            for group=1:numel(stats{layer_id}{batch}) % collect for groups
                means(batch,group) = gather(stats{layer_id}{batch}{group}.mn);
                stds(batch,group) = gather(stats{layer_id}{batch}{group}.sd);
                lcn_means(batch,group) = gather(stats{layer_id}{batch}{group}.lcn_mn);
                feat_means(batch,group) = gather(stats{layer_id}{batch}{group}.mean);
                feat_mins(batch,group) = gather(stats{layer_id}{batch}{group}.min);
                feat_maxs(batch,group) = gather(stats{layer_id}{batch}{group}.max);
            end
        end
        stats{layer_id} = struct('mn', mean(means(:)), 'sd', mean(stds(:)), 'lcn_mn', mean(lcn_means),...
            'feat_mean', mean(feat_means(:)), 'feat_min', mean(feat_mins(:)), 'feat_max', mean(feat_maxs(:)));
        fprintf('layer %d: mn = %f, sd = %f, lcn_mn = %f, feat_mean = %f, feat_min = %f, feat_max = %f \n', ...
            layer_id, stats{layer_id}.mn, stats{layer_id}.sd, mean(stats{layer_id}.lcn_mn), ...
            stats{layer_id}.feat_mean, stats{layer_id}.feat_min, stats{layer_id}.feat_max);
        stats{layer_id}.output_size = stats_size{layer_id}{1}.output_size{1};
    end
end

time_global = toc(time_global);
fprintf('total time: %3.2f sec, avg multilayer speed: %3.2f samples/sec \n', time_global, size(fmaps_out,1)/time_global);

end

% Processes single batch
function [fmaps_out, stats] = forward_pass_batch(fmaps, filters, opts)
% fmaps is a 2d array: n_samples x variables
% it can be reshaped into a 4d array: n_samples x rows x cols x N_filters_{l-1}

n_samples = size(fmaps,1);
opts.sample_size(opts.sample_size == 1) = [];
fmaps = reshape(fmaps, [n_samples, opts.sample_size]);
fmaps = fmaps(:,:,:,~opts.redundant_features);
fmaps = permute(fmaps, [2,3,4,1]); % make suitable for matconvnet
sz_fmaps = size(fmaps);
sz_filters = size(filters);
if (length(sz_filters) < 5), sz_filters(5) = 1; end; % for generalization

if (opts.n_groups == 1 && opts.flip) % layer 1
    fmaps = fliplr(fmaps);
end

% we prefer to do padding here (before feature scaling) instead of in vl_nnconv 
% For MNIST it is good because image values are zeros on the boundaries and, 
% therefore, there is no edge effect
if (any(sz_fmaps(1:2) > sz_filters(1:2)) && opts.is_vl)
    % zero padding if not a fully connected layer
    fmaps = padarray(fmaps, opts.conv_pad, 0, 'both'); 
end
if (~opts.is_vl)
    fmaps = padarray(fmaps, opts.filter_size(1:2) - 1 + opts.conv_pad - ...
            floor(opts.filter_size(1:2)./2),0, 'post');
end
% PREPROCESSING
% treat the entire batch with all feature map groups as a single vector
stats = [];
if (~isempty(opts.stats))
    fmaps = feature_scaling(fmaps, 'stat', opts.stats.mn(1), opts.stats.sd(1));
else
    [fmaps, stats{1}] = feature_scaling(fmaps, 'stat', []);
end

% if (opts.gpu), fmaps = gpuArray(fmaps); end % send to a GPU in the loop in case of the out of memory exception

if (~opts.is_vl)
    for d=opts.dims, fmaps = fft(fmaps,[],d); end
end

fmaps_out = cell(2,opts.n_groups);
% Process in a loop because of overlapping connections (i.e. opts.n_groups*sz_filters(4) > sz_fmaps(3)) and
% without the loop we have to use smaller (less efficient) batch_size to fit into GPU memory
% This can be, probably, done faster
for group=1:opts.n_groups
    
%     Divide feature map into groups according to connections
%     for now connections are binarized (hard), but, in general, they can be soft
%     fmaps_out{group} = bsxfun(@times, fmaps, permute(opts.connections(group,:),[3,1,2]));
%     fmaps_out{group} = fmaps(:,:,sum(sum(sum(abs(fmaps_out{group}),1),2),4) > 1e-10,:);

    % CONVOLUTION
    fmaps_out{1,group} = conv_wrap(fmaps(:,:,opts.connections(group,:),:), filters(:,:,:,:,min(group,sz_filters(5))), opts);
    % fmaps_out{group} is a 4d array: rows x cols x sz_filters(4) x n_samples

    % RECTIFICATION
    if (strcmpi(opts.rectifier,'abs'))
        fmaps_out{1,group} = abs(fmaps_out{1,group});
    elseif (strcmpi(opts.rectifier,'relu'))
        if (opts.is_vl)
            fmaps_out{1,group} = vl_nnrelu(real(fmaps_out{1,group}), [], 'leak', opts.rectifier_leak);
        else
            fmaps_out{1,group} = real(fmaps_out{1,group}); % leak is ignored
        end
    elseif (strcmpi(opts.rectifier,'logistic'))
        k = 0.4;
        fmaps_out{1,group} = 1./(1+exp(-k.*real(fmaps_out{1,group})));
    else
        error('not supported rectifier type')
    end

    if (isempty(opts.stats))
        stats{group} = stats{1};
        stats{group}.max = gather(max(fmaps_out{1,group}(:)));
        stats{group}.min = gather(min(fmaps_out{1,group}(:)));
        stats{group}.mean = gather(mean(fmaps_out{1,group}(:)));
    end

    % Actual ReLU is here
    fmaps_out{1,group} = max(opts.rectifier_param(1), min(opts.rectifier_param(2), fmaps_out{1,group}));
    % the first cell {1,group} is the features that will be passed to the next layers

    % POOLING with larger pooling size for the multidictionary features forward passed directly to a classifier
    if (opts.multidict)
        % the second cell {2,group} is the features that will be passed to a classifier (or PCA)
        pool_size = opts.pool_size;
        opts.pool_size = opts.pool_size_multidict;
        if (opts.pruned)
            % only supported for a 2 layer network
            fmaps_out{2,group} = fmaps_out{1,group}(:,:,sum(opts.connections_next,1) > max(opts.connections_next(:))*1e-5,:);
            fmaps_out{2,group} = pool_wrap(fmaps_out{2,group}, opts); 
        else
            fmaps_out{2,group} = pool_wrap(fmaps_out{1,group}, opts); % propagate all features
        end
        opts.pool_size = pool_size;
    end

    % Local contrast normalization (LCN) for the features forward passed to the next layer
    if (opts.lcn)
        if (isempty(opts.connections_next))
            connections = true(size(fmaps_out{1,group},3),1); % LCN for all feature maps
        else
            % LCN only for those features connected to the next layer
            connections = sum(opts.connections_next,1) > max(opts.connections_next(:))*1e-5;
        end
        if (nnz(connections) <= 1), error('connections are invalid'); end
        if (~isempty(opts.stats))
            fmaps_out{1,group}(:,:,connections,:) = lcn(fmaps_out{1,group}(:,:,connections,:), opts.stats.lcn_mn(group), opts.is_vl); 
        else
            [fmaps_out{1,group}(:,:,connections,:), stats{group}.lcn_mn] = lcn(fmaps_out{1,group}(:,:,connections,:), [], opts.is_vl);
        end
    else
        stats{group}.lcn_mn = nan;
    end
    % POOLING (regular)
    fmaps_out{1,group} = pool_wrap(fmaps_out{1,group}, opts); 
end

% Concatenate all groups
for k=1:size(fmaps_out,1), fmaps_out{k,1} = cat(3,fmaps_out{k,:}); end % 4d array: rows x cols x sz_filters(4)*opts.n_groups x n_samples
fmaps_out = fmaps_out(:,1);
fmaps_out(cellfun(@isempty,fmaps_out)) = [];
if (isempty(opts.stats))
    stats{1}.output_size = cellfun(@size,fmaps_out,'UniformOutput',false);
end
% reshape features back to vectors
for k=1:size(fmaps_out,1)
%     if (opts.gpu), fmaps_out{k,1} = gather(fmaps_out{k,1});  end
    sz_ouput = size(fmaps_out{k,1});  % a 4d array: rows x cols x sz_filters(4)*opts.n_groups x n_samples
    fmaps_out{k,1} = reshape(fmaps_out{k,1}, [prod(sz_ouput(1:3)), n_samples])'; % sz_ouput(4) can cause an error
    % fmaps_out{k,1} is a 2d array: n_samples x prod(sz_ouput(1:3))
end

end

% Convolution wrapper for convenience
function fmaps = conv_wrap(fmaps, filters, opts)
if (opts.is_vl)
    % using Matconvnet
    if (isreal(filters))
        fmaps = vl_nnconv(fmaps, filters, [], 'stride', opts.conv_stride);
    else
        fmaps = vl_nnconv(fmaps, real(filters), [], 'stride', opts.conv_stride) + 1i.*vl_nnconv(fmaps, imag(filters), [], 'stride', opts.conv_stride);
    end
else
    % using Matlab in the frequency domain
    fmaps = bsxfun(@times, permute(fmaps,[1:3,5,4]), filters);
    for d=opts.dims, fmaps = ifft(fmaps,[],d); end
    sz = size(fmaps);
    offset = floor((sz(1:2) - opts.sample_size(1:2))./2);
    fmaps = squeeze(sum(fmaps(offset(1)+1:end-offset(1),offset(2)+1:end-offset(2),:,:,:),3));
end

end

% Pooling wrapper for convenience
function fmaps = pool_wrap(fmaps, opts)
if (opts.pool_size <= 1)
    return;
end
if (isfield('opts','pool_fn') && ~isempty(opts.pool_fn))
    fmaps = opts.pool_fn(fmaps, opts); % can be some custom pooling function
elseif (~opts.is_vl)
    fmaps = pool_disjoint(fmaps, opts.pool_size, opts.pool_pad, opts.pool_op);
else
    fmaps = vl_nnpool(fmaps, opts.pool_size, 'stride', opts.pool_size, 'pad', opts.pool_pad, 'method', opts.pool_op);
end

end

% Pooling from squared disjoint regions within a feature map
function fmaps = pool_disjoint(fmaps, pool_size, pool_pad, method)
if (ischar(method))
    if (strcmpi(method,'max'))
        pool_op = @(input) max(max(input,[],1),[],2);
    elseif (strcmpi(method,'avg'))
        pool_op = @(input) mean(mean(input,1),2);
    else
        error('not supported pooling method')
    end
else
    pool_op = method;
end
if (pool_pad > 0)
    fmaps = padarray(fmaps, [pool_pad,pool_pad], 0, 'both');
end
for c=1:size(fmaps,2)/pool_size
    for r=1:size(fmaps,1)/pool_size
        fmaps(pool_size*(r-1)+1,pool_size*(c-1)+1,:,:) = ...
            pool_op(fmaps(pool_size*(r-1)+1:pool_size*r,pool_size*(c-1)+1:pool_size*c,:,:));
    end
end
fmaps = fmaps(1:pool_size:end,1:pool_size:end,:,:);
end

% Normalizes feature maps (for MNIST)
function fmaps = local_fmaps_norm(fmaps, sample_size)
feat_norm = 'l2';
% fprintf('local feature maps %s-scaling \n', feat_norm)
sz = size(fmaps); % 2d array: n_samples x features
n = prod(sample_size(1:2));
fmaps = permute(reshape(fmaps, round([sz(1), n, sz(end)/n])), [1,3,2]); % 3d array: n_samples x n_filters x n_pixels 
fmaps = reshape(fmaps, round([sz(1)*sz(end)/n, n])); % 2d array: n_samples*n_filters x n_pixels
if (isa(fmaps,'gpuArray'))
    fmaps = gpuArray(feature_scaling(gather(fmaps), feat_norm));
else
    fmaps = feature_scaling(fmaps, feat_norm);
end
% reshape back to vectors
fmaps = reshape(permute(reshape(fmaps, [sz(1),sz(end)/n,n]),[1,3,2]),[sz(1),n,sz(end)/n]);
fmaps = reshape(fmaps,sz);

end

% Normalizes filters
function filters = norm_5d(filters)
for group=1:size(filters,5)
    for k=1:size(filters,4)
        f = filters(:,:,:,k,group);
        if (std(f(:)) < 1e-10)
            error('filter might be blank')
        end
        filters(:,:,:,k,group) = f./norm(f(:));
    end
end

end

% Hilbert transform to filters
function filters = hilbert_5d(filters)
for group=1:size(filters,5)
    for k=1:size(filters,4)
        f = filters(:,:,:,k,group);
        filters(:,:,:,k,group) = reshape(hilbert(f(:)),size(f));
    end
end

end

% PCA + whitening
function features = pca_whiten_wrap(features, opts)
if (isfield(opts,'PCA_matrix') && ~isempty(opts.PCA_matrix))
    opts.verbose = false;
    opts.pca_mode = 'pcawhiten';
    if (iscell(opts.PCA_matrix))
        sz = [opts.sample_size(1:2)./opts.pool_size, opts.n_filters, opts.n_groups];
        features_reshaped = reshape(features(:,1:prod(sz)), [size(features,1), sz]);
        features_split = cell(1,opts.n_groups);
        for group = 1:opts.n_groups
            features_split{group} = ...
                pca_zca_whiten(reshape(features_reshaped(:,:,:,:,group), [size(features_reshaped,1), ...
                prod(sz(1:2))*opts.n_filters]), opts, opts.PCA_matrix{group}, opts.data_mean{group}, opts.L_regul{group});
        end
        % normalize features, concatenate with lower layer features
        features_split = feature_scaling(cat(2,features_split{:}), 'stat');
        features = feature_scaling(cat(2,features_split,feature_scaling(features(:,prod(sz)+1:end),'stat')), 'stat');
    else
        features = pca_zca_whiten(features, opts, opts.PCA_matrix, opts.data_mean, opts.L_regul);
    end
end

end

function opts = set_default_values(opts)
if (~isfield(opts,'progress_print'))
    opts.progress_print = 10; % print statistics every 10th batch
end
if (~isfield(opts,'is_vl'))
    % true to use Matconvnet (faster, about 4-6 times in my case), otherwise use Matlab implementation
    % although, with Matlab implementation I consistently get slightly better classification accuracy
    % due to some GPU issue, changing this option can cause a GPU error
    opts.is_vl = true; 
end
if (~isfield(opts,'conv_stride'))
    opts.conv_stride = 1; % convolution stride
end
if (~isfield(opts,'rectifier'))
    opts.rectifier = 'relu';
end
if (~isfield(opts,'rectifier_param'))
    opts.rectifier_param = [0,Inf]; % ReLU
end
if (~isfield(opts,'rectifier_leak'))
    opts.rectifier_leak = 0; % Leaky ReLU
end
if (~isfield(opts,'pool_size'))
    opts.pool_size = 2; % pooling size
end
if (~isfield(opts,'pool_op'))
    opts.pool_op = 'max'; % pooling type
end
if (~isfield(opts,'multidict'))
    opts.multidict = true; % true to use feature maps of all layers
end
if (~isfield(opts,'pruned'))
    % only feature maps connected to the next layer are used as multidictionary features
    opts.pruned = true;
end
if (~isfield(opts,'stats'))
    opts.stats = []; % statistical data of batches
end
if (~isfield(opts,'complex_filters'))
    opts.complex_filters = false; % complex valued filters
end
if (~isfield(opts,'flip'))
    opts.flip = false; % true to flip input images horizontally
end

end