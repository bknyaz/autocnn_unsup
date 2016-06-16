function net = net_init(arch, varargin)
% Initializes the network and its parameters
% Example usage:
% net = net_init('1024c13-2p-conv0_4__128g-4ch-160c11-4p-conv2_3', 'sample_size', [32,32,1,3], 'batch_size', 128)
% varargin{1} can be a structure with parameters which are added to each layer
% varargin can contain pairs <'field_name', value>, which override values in varargin{1}
% sample_size is a mandatory field

if (isstruct(varargin{1}))
    net = parse_net_arch(arch, varargin{1});
else
    net = parse_net_arch(arch, []);
end

for layer_id=1:numel(net.layers)
    
    if (layer_id == 1)
        % size of inputs to layer 1
        net.layers{layer_id}.sample_size = find_value(varargin, 'sample_size', layer_id, []);
    else
        % size of inputs to layer > 1
        net.layers{layer_id}.sample_size = [net.layers{layer_id-1}.sample_size(1:2)./net.layers{layer_id-1}.pool_size,...
            1,net.layers{layer_id-1}.n_groups*net.layers{layer_id-1}.n_filters];
    end
    
    % filter response normalization
    net.layers{layer_id}.norm = find_value(varargin, 'conv_norm', layer_id, 'stat');
    if (strcmpi(net.layers{layer_id}.norm,'rootsift'))
        net.layers{layer_id}.norm_pow = 0.5; % power for rootsift normalization (otherwise ignored)
    end
    
    % number of samples in a batchs
    net.layers{layer_id}.batch_size = find_value(varargin, 'batch_size', layer_id, 128);
    
    % Use a GPU
    net.layers{layer_id}.gpu = find_value(varargin, 'gpu', layer_id, true);
    
    % rectifier
    net.layers{layer_id}.rectifier = find_value(varargin, 'rectifier', layer_id, 'relu');
    net.layers{layer_id}.rectifier_param = find_value(varargin, 'rectifier_param', layer_id, [0,Inf]);
    
    % method to learn filters
    net.layers{layer_id}.learning_method = find_value(varargin, 'learning_method', layer_id, 'kmeans');
    net.layers{layer_id}.norm_type = layer_id; % method to normalize autoconvolutional responses
    net.layers{layer_id}.shared_filters = find_value(varargin, 'shared_filters', layer_id, true);
    
    % Turn local contrast normalization on/off
    lcn = find_value(varargin, 'lcn', layer_id, -1);
    if (lcn > 0)
        net.layers{layer_id}.lcn = lcn;
    end
        
    % pooling
    net.layers{layer_id}.pool_op = find_value(varargin, 'pool_op', layer_id, 'max');

    % zero-padding for convolution
    net.layers{layer_id}.conv_pad = find_value(varargin, 'conv_pad', layer_id, floor(net.layers{layer_id}.filter_size./2));
    
    % use features from all layers
    net.layers{layer_id}.multidict = find_value(varargin, 'multidict', layer_id, true);
    
    if (net.layers{layer_id}.multidict && layer_id < numel(net.layers))
        pool_size_multidict = 1;
        for l=layer_id:numel(net.layers)
            pool_size_multidict = pool_size_multidict*net.layers{l}.pool_size;
        end
        net.layers{layer_id}.pool_size_multidict = pool_size_multidict;
    end
    
    % use data augmentation
    net.layers{layer_id}.augment = find_value(varargin, 'augment', layer_id, false);
    
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
    for b=1:numel(blocks)
        if (strfind(blocks{b},'conv'))
            id = strfind(blocks{b},'conv');
            if (length(blocks{b}) >= 7)
                net.layers{l}.conv_orders = str2double(blocks{b}(id(1)+4)):str2double(blocks{b}(id(1)+6));
            else
                net.layers{l}.conv_orders = str2double(blocks{b}(id(1)+4));
            end
        elseif (strfind(blocks{b},'ch'))
            net.layers{l}.n_channels_max = str2double(blocks{b}(1:end-2));
        elseif (strfind(blocks{b},'c'))
            id = strfind(blocks{b},'c');
            net.layers{l}.n_filters = str2double(blocks{b}(1:id(1)-1));
            net.layers{l}.filter_size = repmat(str2double(blocks{b}(id(1)+1:end)),1,2);
        elseif (strfind(blocks{b},'p'))
            net.layers{l}.pool_size = str2double(blocks{b}(1:end-1));
        elseif (strfind(blocks{b},'g'))
            net.layers{l}.n_groups = str2double(blocks{b}(1:end-1));
        end
    end
end

end

function value = find_value(pairs, query, layer_id, default_value)

value = default_value;

for p=1:numel(pairs)
    if (ischar(pairs{p}) && strcmpi(pairs{p},query))
        value = pairs{p+1};
        break;
    end
end

if (iscell(value))
    value = value{layer_id};
end

if (isempty(value))
    error('parameter not found nor specified')
end

end