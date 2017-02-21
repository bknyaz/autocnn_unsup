function net = net_init(arch, varargin)
% Initializes the network and its parameters
% Example usage:
% net = net_init('1024c13-2p-conv0_4__128g-4ch-160c11-4p-conv2_3', 'sample_size', [32,32,3], 'batch_size', 128)
% varargin{1} can be a structure with parameters which are added to each layer
% varargin can contain pairs <'field_name', value>, which override values in varargin{1}
% sample_size is a mandatory field

if (isstruct(varargin{1}))
    net = parse_net_arch(arch, varargin{1});
else
    net = parse_net_arch(arch, []);
end

for layer_id=1:numel(net.layers)
    
    % size of inputs to layer 1
    if (layer_id == 1)
        net.layers{layer_id}.sample_size = find_value(varargin, 'sample_size', layer_id, []);
        if (length(net.layers{layer_id}.sample_size) == 3)
            net.layers{layer_id}.sample_size = [net.layers{layer_id}.sample_size(1:2),1,net.layers{layer_id}.sample_size(3)];
        end
        net.layers{layer_id}.filter_size(3) = net.layers{layer_id}.sample_size(end);
    end
    
    net.layers{layer_id}.crop = find_value(varargin, 'crop', layer_id, 0);
    
    % display progress in detail
    net.layers{layer_id}.verbose = find_value(varargin, 'verbose', layer_id, true);
    
    % filter response normalization
    net.layers{layer_id}.norm = find_value(varargin, 'conv_norm', layer_id, 'stat');
    if (isfield(net.layers{layer_id},'conv_norm'))
        net.layers{layer_id} = rmfield(net.layers{layer_id}, 'conv_norm');
    end
    
    % number of samples in a batchs (same for all layers)
    net.layers{layer_id}.batch_size = find_value(varargin, 'batch_size', layer_id, 100);
    
    % Use a GPU
    net.layers{layer_id}.gpu = find_value(varargin, 'gpu', layer_id, true);
    
    % rectifier
    net.layers{layer_id}.rectifier = find_value(varargin, 'rectifier', layer_id, 'relu');
    net.layers{layer_id}.rectifier_param = find_value(varargin, 'rectifier_param', layer_id, [0,Inf]);
    
    % method to learn filters
    net.layers{layer_id}.learning_method = find_value(varargin, 'learning_method', layer_id, 'kmeans');
    net.layers{layer_id}.norm_type = layer_id; % method to normalize autoconvolutional responses
    if (layer_id > 1)
        net.layers{layer_id}.shared_filters = find_value(varargin, 'shared_filters', layer_id, true);
        net.layers{layer_id}.connections_complete = find_value(varargin, 'connections_complete', layer_id, true);
    end
    
    % Turn local contrast normalization on/off
    lcn = find_value(varargin, 'lcn', layer_id, -1);
    if (lcn >= 0)
        net.layers{layer_id}.lcn = lcn;
    end
        
    % pooling
    net.layers{layer_id}.pool_op = find_value(varargin, 'pool_op', layer_id, 'max');

    % zero-padding for convolution
    net.layers{layer_id}.conv_pad = find_value(varargin, 'conv_pad', layer_id, floor(net.layers{layer_id}.filter_size(1:2)./2));
    
    % use features from all layers
    if (numel(net.layers) > 1)
        
        
        net.layers{layer_id}.pool_pad = 1;
        
        net.layers{layer_id}.multidict = find_value(varargin, 'multidict', layer_id, true);
        if (layer_id > 1)
            net.layers{layer_id}.pruned = false;
        end
    end
    
    if (layer_id < numel(net.layers) && net.layers{layer_id}.multidict)
        pool_size_multidict = 1;
        for l=layer_id+1:numel(net.layers)
            pool_size_multidict = pool_size_multidict*net.layers{l}.pool_stride;
        end
        net.layers{layer_id}.pool_size_multidict = pool_size_multidict;
    end
    
    % use data augmentation
    if (layer_id == 1)
        net.layers{layer_id}.augment = find_value(varargin, 'augment', layer_id, false);
    end
    
    % Print properties
    fprintf('Layer %d \n', layer_id)
    net.layers{layer_id}
end

end

function net = parse_net_arch(arch, opts)
% Network architecture parser
% e.g., arch = '256c13-4p-conv0_3__64g-2ch-128c9-2p-conv2_3' or
% arch = '1024c13-2p-conv0_4__128g-4ch-160c11-4p-conv2_3'
% opts - general options which will be added to each layer

arch = strtrim(lower(arch));
layers = strsplit(arch,'__');
layers(cellfun(@isempty,layers)) = [];
n_layers = numel(layers);
net.layers = cell(1,n_layers);
net.arch = arch;

for l=1:n_layers
    
    net.layers{l} = opts;
    net.layers{l}.n_groups = 1;
    
    blocks = strsplit(layers{l},'-');
    filter_depth = 1;
    for b=1:numel(blocks)
        if (strfind(blocks{b},'conv'))
            id = strfind(blocks{b},'conv');
            net.layers{l}.conv_orders = str2double(blocks{b}(id(1)+4));
            if (length(blocks{b}) >= 7)
              b2 = str2double(blocks{b}(id(1)+6));
              if (isfinite(b2))
                net.layers{l}.conv_orders = net.layers{l}.conv_orders:b2;
              end
            end
        elseif (strfind(blocks{b},'ch'))
            filter_depth = str2double(blocks{b}(1:end-2));
        elseif (strfind(blocks{b},'c'))
            id = strfind(blocks{b},'c');
            net.layers{l}.n_filters = str2double(blocks{b}(1:id(1)-1));
            net.layers{l}.filter_size = repmat(str2double(blocks{b}(id(1)+1:end)),1,2);
        elseif (strfind(blocks{b},'p'))
            net.layers{l}.pool_size = str2double(blocks{b}(1:end-1));

        elseif (strfind(blocks{b},'s'))
            net.layers{l}.pool_stride = str2double(blocks{b}(1:end-1));
        elseif (strfind(blocks{b},'g'))
            net.layers{l}.n_groups = str2double(blocks{b}(1:end-1));
        end
    end
    net.layers{l}.filter_size = [net.layers{l}.filter_size,filter_depth];
    if (~isfield(net.layers{l},'pool_stride'))
        if (n_layers > 1)
            net.layers{l}.pool_stride = net.layers{l}.pool_size-1;
        else
            net.layers{l}.pool_stride = net.layers{l}.pool_size;
        end
    end
end

end

function value = find_value(pairs, query, layer_id, default_value)

value = default_value;

for p=1:numel(pairs)
    if (isstruct(pairs{p}) && isfield(pairs{p},query))
        value = pairs{p}.(query);
        % no break here
    elseif (ischar(pairs{p}) && strcmpi(pairs{p},query))
        value = pairs{p+1};
        break;
    end
end

if (iscell(value))
    value = value{layer_id};
end

end