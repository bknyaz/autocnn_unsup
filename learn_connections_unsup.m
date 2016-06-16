function connections = learn_connections_unsup(feature_maps, opts)
% Learns connections from layer l to l+1 in the unsupervised way
% This procedure is adopted from paper "Selecting Receptive Fields in Deep Networks" by A. Coates and A.Y. Ng
%
% feature_maps - feature maps of layer l (input samples to layer l+1), rows are samples, columns are variables, 
% there are prod(opts.sample_size) columns
% opts - a structure with parameters of layer l+1
% connections - a binary connection matrix with opts.n_groups rows and N_filters_l columns

if (opts.n_groups*opts.filter_size(3) == opts.sample_size(end))
    % create simple sequential connections in case of a complete connection scheme
    connections = reshape(1:opts.sample_size(end),opts.filter_size(3),opts.n_groups)';
    fprintf('complete connection scheme is applied \n')
else
    opts.pca_mode = 'zcawhiten';
    opts.pca_epsilon = 0.05;
    opts.pca_fast = 1;
    opts.pca_dim = [];

    % reshape features
    feature_maps = reshape(feature_maps,[size(feature_maps,1),opts.sample_size]);
    feature_maps = reshape(feature_maps,[size(feature_maps,1)*prod(opts.sample_size(1:2)),opts.sample_size(end)]);
    feature_maps = feature_maps(randperm(size(feature_maps,1),min(size(feature_maps,1),10^5)),:);
    feature_maps = feature_scaling(feature_maps, 'gray'); % normalize feature maps in range [0,1]
    feature_maps = pca_zca_whiten(feature_maps, opts); % whiten feature maps
    feature_maps = (feature_maps.^2)';
    % correlation distance between all pairs of feature maps
    S_all = mat2gray(pdist2(feature_maps,feature_maps,'correlation'));
    n_it = 100; % repeat random selection n_it times
    connections_all = cell(1,n_it);
    S_total = zeros(1,n_it);
    for it=1:n_it
        S_all_it = S_all;
        rand_feats = randperm(size(S_all_it,1),min(size(S_all_it,1),opts.n_groups)); % select opts.n_groups random feature maps
        connections_tmp = [];
        for i=1:opts.n_groups
            while (true)
                S_tmp = S_all_it;
                S_tmp(rand_feats(i),rand_feats(i)) = Inf; % prevent selection of the same feature map
                [S_min,id] = sort(S_tmp(rand_feats(i),:)); % find the closest feature maps
                closest = S_min(1:opts.filter_size(3)-1); 
                group = sort([rand_feats(i),id(1:opts.filter_size(3)-1)]); % group of feature maps
                cc_tmp = cat(1,connections_tmp,group);
                % select only unique groups and try to select the groups that connect more feature maps
                if (length(unique(cc_tmp(:,1))) == size(cc_tmp,1)) 
                    break;
                end
                rand_feats(i) = randperm(size(S_all_it,1),1); % try another feature map
            end
            S_all_it = S_tmp;
            connections_tmp(i,:) = group; % add a selected connection
            S_total(it) = S_total(it)+sum(closest); % count total sum of distances
        end
        connections_all{it} = connections_tmp;
    end
    [~,id] = sort(S_total); % find iterations with minimum total sum of distances
    % obtain the number of unique feature maps connected to layer l+1
    n_unique_feats = zeros(1,round(n_it/10));
    for it=1:round(n_it/10)
      n_unique_feats(it) = length(unique(connections_all{id(it)}));
    end
    % among the iterations with the minimum total sum of distances 
    % find the iteration with the maximum total number of unique feature maps
    it = id(find(n_unique_feats == max(n_unique_feats),1,'first'));
    connections = unique(connections_all{it},'rows');

end

% convert connections to a binary format
connections_bin = false(opts.n_groups, opts.sample_size(end));
for c=1:size(connections_bin,1)
    connections_bin(c,connections(c,:)) = true;
end
connections = connections_bin;

end