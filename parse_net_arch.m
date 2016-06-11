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
                net.layers{l}.conv_orders = {[str2double(blocks{b}(id(1)+4)):str2double(blocks{b}(id(1)+6))]};
            else
                net.layers{l}.conv_orders = {[str2double(blocks{b}(id(1)+4))]};
            end
        elseif (strfind(blocks{b},'ch'))
            net.layers{l}.n_channels_max = str2double(blocks{b}(1:end-2));
        elseif (strfind(blocks{b},'c'))
            id = strfind(blocks{b},'c');
            net.layers{l}.n_filters = str2double(blocks{b}(1:id(1)-1));
            net.layers{l}.filter_size = str2double(blocks{b}(id(1)+1:end));
        elseif (strfind(blocks{b},'p'))
            net.layers{l}.pool_size = str2double(blocks{b}(1:end-1));
        elseif (strfind(blocks{b},'g'))
            net.layers{l}.n_groups = str2double(blocks{b}(1:end-1));
        end
    end
end

end