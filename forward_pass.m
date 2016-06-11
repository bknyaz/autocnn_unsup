function [feature_maps, stats] = forward_pass(feature_maps, net)
% Processes samples (images) 'feature_maps' according to the multilayer architecture of network 'net'
% Uses Matconvnet for faster convolutions and other standard operations (e.g., pooling)
% 
% feature_maps - a matrix, in which rows are samples and columns are variables (features)
% stats - statistics obtained from the training samples later used for the test samples
% net.layers - a cell array of layer parameters
% Each layer contains filters (a 5d array) and connections (a 2d array) and stats (in case of test samples),
% other model parameters are fixed (not learned)

% init variables
time = tic;
n_layers = numel(net.layers);
stats = cell(1,n_layers);
feature_maps_multi = cell(1,n_layers);

% We process samples layer-wise in opposite to traditional batch-wise,
% so, first, all samples are processed with the first layer filters, then with the second layer filters, etc.
% It has some advantages, e.g., we have to send filters to a GPU only for the current layer
for layer_id = 1:n_layers
    
    sz_filters = size(net.layers{layer_id}.filters);
    if (length(sz_filters) < 5), sz_filters(5) = 1; end;
    fprintf('-> %s %d: %d feature maps from layer %d used, %d groups, filters %dx%dx%dx%dx%d \n', ...
        upper('layer'), layer_id, nnz(sum(net.layers{layer_id}.connections,1) > 1e-10), layer_id-1, ...
        size(net.layers{layer_id}.connections,1), sz_filters)
    
    if (layer_id > 1 && isfield(net.layers{layer_id},'lcn_l2') && net.layers{layer_id}.lcn_l2)
        % useful for MNIST: scale feature maps before passing to the next layer
        feature_maps = local_fmaps_norm(feature_maps, net.layers{layer_id}.sample_size);
    end
    
    % for the last layer multidictionary features are not applicable
    if (layer_id == n_layers), net.layers{layer_id}.multidict = false; end % override the value
    
    if (~isfield(net.layers{layer_id},'connections_next') && layer_id < n_layers)
        net.layers{layer_id}.connections_next = net.layers{layer_id+1}.connections; % to prune features
    else
        net.layers{layer_id}.connections_next = [];
    end
    
    if (~isfield(net.layers{layer_id},'lcn'))
        % for the last layer local contrast normalization is not useful
        net.layers{layer_id}.lcn = layer_id < n_layers;
    end
    
    % process all samples according to the architecture of layer l
    [feature_maps, feature_maps_multi{layer_id}, stats{layer_id}] = forward_pass_layer(feature_maps, ...
        net.layers{layer_id}.filters, net.layers{layer_id});
    
    % apply feature normalization for each sample
    if (~isempty(net.layers{layer_id}.norm))
        feature_maps = feature_scaling(feature_maps, net.layers{layer_id}.norm);
        if (net.layers{layer_id}.multidict)
            feature_maps_multi{layer_id} = feature_scaling(feature_maps_multi{layer_id}, net.layers{layer_id}.norm);
        end
    end
    
    % collect feature maps statistics from the training samples
    if (isfield(net.layers{layer_id},'stats') && ~isempty(net.layers{layer_id}.stats))
        continue;
    end
    fprintf('collecting statistics for layer %d \n', layer_id)
    means = []; % mean values of all batches
    stds = []; % std values of all batches
    lcn_means = []; % LCN weighted standard deviations
    for batch=1:numel(stats{layer_id}) % collect for batches
        for group=1:numel(stats{layer_id}{batch}) % collect for groups
            means(batch,group) = stats{layer_id}{batch}{group}.mn;
            stds(batch,group) = stats{layer_id}{batch}{group}.sd;
            lcn_means(batch,group) = stats{layer_id}{batch}{group}.lcn_mn;
        end
    end
    stats{layer_id} = struct('mn', mean(means), 'sd', mean(stds), 'lcn_mn', mean(lcn_means));
    fprintf('layer %d: mn = %f, sd = %f, lcn_mn = %f \n', layer_id, stats{layer_id}.mn, stats{layer_id}.sd, stats{layer_id}.lcn_mn);
end

% concatenate features from all layers
if (net.layers{1}.multidict)
    for layer_id=1:numel(net.layers)-1
        feature_maps = cat(2,feature_maps,feature_maps_multi{layer_id});
    end
    % normalize concatenated features
    feature_maps = feature_scaling(feature_maps, net.layers{end}.norm);
end

time = toc(time);
fprintf('avg multilayer speed: %3.3f samples/sec \n', size(feature_maps,1)/time);

end

% Processes single layer (helper wrapper function)
function [fmaps_out, fmaps_out_multi, stats] = forward_pass_layer(fmaps, filters, opts)

% set default values
opts = set_default_values(opts);
if (~isfield('opts','conv_pad'))
    opts.conv_pad = floor([size(filters,1),size(filters,2)]./2); % zero padding for convolution
end

filters = norm_5d(filters); % normalize filters
if (~isempty(opts.gpu))
    filters = gpuArray(filters);
end

% Reorganize feature maps according to the connection matrix for memory efficiency
% the connection matrix has opts.n_groups rows and the number of columns
% equals the number of filters in the previous layer, i.e. N_filters_{l-1}
% filters is a 5d array: rows x cols x depth x N_filters x n_groups, although depth and n_groups can be 1
% feature_maps is a 2d array, which can be reshaped into a 4d array: n_samples x rows x cols x N_filters_{l-1}
redundant_features = sum(opts.connections,1) < max(opts.connections(:))*1e-5;
opts.connections = opts.connections(:,~redundant_features);
opts.redundant_features = redundant_features;

n_samples = size(fmaps,1);
n_batches = ceil(n_samples/opts.batch_size);

% determine the size of feature_maps_out and preallocate an array accordingly
% print checksum for the first 2 samples
for sample_id=1:min(2,n_samples)
    features = forward_pass_batch(fmaps(sample_id,:), filters, opts);
    fprintf('checksum for sample %d = %f \n', sample_id, norm(features{1}(:)));
end
fmaps_out = zeros(n_samples, numel(features{1}), 'single');
fprintf('-> feature maps: %dx%d (%s) \n', size(fmaps_out), class(fmaps_out))
if (opts.multidict)
    fmaps_out_multi = zeros(n_samples, numel(features{2}), 'single');
    fprintf('-> Multidict feature maps: %dx%d (%s) \n', size(fmaps_out_multi), class(fmaps_out_multi))
else
    fmaps_out_multi = [];
end

stats = cell(1,n_batches);
opts.batch_size = min(n_samples, opts.batch_size);

for batch_id = 1:n_batches
    time = tic; % to measure forward pass speed
    samples_ids = max(1,min(n_samples, (batch_id-1)*opts.batch_size+1:1:batch_id*opts.batch_size));
    [features, stats{batch_id}] = forward_pass_batch(fmaps(samples_ids,:), filters, opts);
    fmaps_out(samples_ids,:) = features{1};
    if (opts.multidict)
        fmaps_out_multi(samples_ids,:) = features{2};
    end
    time = toc(time);
    if (mod(batch_id,opts.progress_print)==0), fprintf('batch %d/%d, %3.3f samples/sec \n', batch_id, n_batches, length(samples_ids)/time); end
end

end

% Processes single batch
function [fmaps_out, stats] = forward_pass_batch(fmaps, filters, opts)
% fmaps is a 2d array: n_samples x variables
% it can be reshaped into a 4d array:
% n_samples x rows x cols x N_filters_{l-1}

n_samples = size(fmaps,1);
opts.sample_size(opts.sample_size == 1) = [];
fmaps = reshape(fmaps, [n_samples, opts.sample_size]);
fmaps = fmaps(:,:,:,~opts.redundant_features);
fmaps = permute(fmaps, [2,3,4,1]); % make suitable for matconvnet
sz_fmaps = size(fmaps);
sz_filters = size(filters);
if (length(sz_filters) < 5), sz_filters(5) = 1; end; % for generalization

% we prefer to do padding here (before feature scaling) instead of in vl_nnconv 
% For MNIST it is good because image values are zeros on the boundaries and, 
% therefore, there is no edge effect
if (any(sz_fmaps(1:2) > sz_filters(1:2)))
    % zero padding if not a fully connected layer
    fmaps = padarray(fmaps, opts.conv_pad, 0, 'both'); 
end

% PREPROCESSING
% treat the entire batch with all feature map groups as a single vector
stats = [];
if (~isempty(opts.stats))
    fmaps = feature_scaling(fmaps, 'stat', opts.stats.mn(1), opts.stats.sd(1));
else
    [fmaps, stats{1}] = feature_scaling(fmaps, 'stat', []);
end

% divide feature map into groups according to connections
fmaps_out = cell(1,opts.n_groups);
for group=1:opts.n_groups
    % for now connections are binarized (hard), but, in general, they can be soft
    fmaps_out{group} = bsxfun(@times, fmaps, permute(opts.connections(group,:),[3,1,2]));
    fmaps_out{group} = fmaps(:,:,sum(sum(sum(abs(fmaps_out{group}),1),2),4) > 1e-10,:);
end

% CONVOLUTION
% Process in a loop because of overlapping connections (i.e. opts.n_groups*sz_filters(4) > sz_fmaps(3)) and
% without the loop we have to use smaller (less efficient) batch_size to fit into GPU memory
% This can be, probably, done faster
for group=1:opts.n_groups
    if (~isempty(opts.gpu)), fmaps_out{group} = gpuArray(fmaps_out{group}); end
    fmaps_out{group} = vl_nnconv(fmaps_out{group}, filters(:,:,:,:,min(group,sz_filters(5))), [], 'stride', opts.conv_stride);
    if (~opts.lcn)
        if (~isempty(opts.gpu)), fmaps_out{group} = gather(fmaps_out{group}); end
    end % otherwise, we will need a GPU array later
    % fmaps_out{group} is a 4d array: rows x cols x sz_filters(4) x n_samples
end

% Concatenate all groups
fmaps_out = cat(3,fmaps_out{:}); % 4d array: rows x cols x sz_filters(4)*opts.n_groups x n_samples

% RECTIFICATION
if (strcmpi(opts.rectifier,'abs'))
    fmaps_out = abs(fmaps_out);
elseif (strcmpi(opts.rectifier,'relu'))
	fmaps_out = vl_nnrelu(real(fmaps_out), [], 'leak', opts.rectifier_leak);
elseif (strcmpi(opts.rectifier,'logistic'))
    k = 0.4;
    fmaps_out = 1./(1+exp(-k.*real(fmaps_out)));
else
    error('not supported rectifier type')
end

fmaps_out = {max(opts.rectifier_param(1), min(opts.rectifier_param(2), fmaps_out))};
% the first cell is the features that will be passed to the next layers
% the second cell is the feauters that will be passed to a classifier (or PCA)

% POOLING with larger pooling size for the multidictionary features forward passed directly to a classifier
if (opts.multidict)
    pool_size = opts.pool_size;
    opts.pool_size = opts.pool_size_multidict;
    if (opts.pruned)
        fmaps_out{2} = fmaps_out{1}(:,:,sum(opts.connections_next,1) > max(opts.connections_next(:))*1e-5,:);
        fmaps_out{2} = pool_wrap(fmaps_out{2}, opts); 
    else 
        fmaps_out{2} = pool_wrap(fmaps_out{1}, opts); % propagate all features
    end
    if (opts.lcn && opts.gpu)
        fmaps_out{2} = gather(fmaps_out{2});
    end
    opts.pool_size = pool_size;
end

% Local contrast normalization (LCN) for the features forward passed to the next layer
if (opts.lcn)
    if (n_samples == 1), fprintf('local contrast normalization \n'); end % print once
    if (isempty(opts.connections_next))
        connections = true(size(fmaps_out{1},3),1); % LCN for all feature maps
    else
        % LCN only for those features connected to the next layer
        connections = sum(opts.connections_next,1) > max(opts.connections_next(:))*1e-5;
    end
    if (nnz(connections) <= 1), error('connections are invalid'); end
    if (~isempty(opts.stats))
        fmaps_out{1}(:,:,connections,:) = lcn(fmaps_out{1}(:,:,connections,:), opts.stats.lcn_mn); 
    else
        [fmaps_out{1}(:,:,connections,:), stats{1}.lcn_mn] = lcn(fmaps_out{1}(:,:,connections,:), []);
    end
else
    stats{1}.lcn_mn = nan;
end

% POOLING (regular)
fmaps_out{1} = pool_wrap(fmaps_out{1}, opts); 

if (opts.lcn && opts.gpu), fmaps_out{1} = gather(fmaps_out{1}); end

% reshape features back to vectors
for k=1:numel(fmaps_out)
    sz_ouput = size(fmaps_out{k});  % a 4d array: rows x cols x sz_filters(4)*opts.n_groups x n_samples
    fmaps_out{k} = reshape(fmaps_out{k}, [prod(sz_ouput(1:3)), n_samples]); % sz_ouput(4) can cause an error
    if (n_samples > 1)
        fmaps_out{k} = fmaps_out{k}'; % a 2d array: n_samples x prod(sz_ouput(1:3))
    end
end

end

% Pooling wrapper for convenience
function fmaps = pool_wrap(fmaps, opts)
if (opts.pool_size <= 1)
    return;
end
if (isfield('opts','pool_fn') && ~isempty(opts.pool_fn))
    fmaps = opts.pool_fn(fmaps, opts); % can be some custom pooling function
else
    fmaps = vl_nnpool(fmaps, opts.pool_size, 'stride', opts.pool_size, 'method', opts.pool_op);
end

end

% Normalizes feature maps (for MNIST)
function fmaps = local_fmaps_norm(fmaps, sample_size)
feat_norm = 'l2';
fprintf('local feature maps %s-scaling \n', feat_norm)
sz = size(fmaps); % 2d array: n_samples x features
n = prod(sample_size(1:2));
fmaps = permute(reshape(fmaps, round([sz(1), n, sz(end)/n])), [1,3,2]); % 3d array: n_samples x n_filters x n_pixels 
fmaps = reshape(fmaps, round([sz(1)*sz(end)/n, n])); % 2d array: n_samples*n_filters x n_pixels
fmaps = feature_scaling(fmaps, feat_norm);
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

function opts = set_default_values(opts)
if (~isfield(opts,'progress_print'))
    opts.progress_print = 10; % print statistics every 10th batch
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
if (opts.multidict)
    if (~isfield(opts,'pruned'))
        % only feature maps connected to the next layer are used as multidictionary features
        opts.pruned = true;
    end
end
if (~isfield(opts,'stats'))
    opts.stats = []; % statistical data of batches
end

end