function [D, ids] = conv_kmeans(W, k, wS, wS_ratio, batch_size, max_iter, vis)
% Implementation of convolutional k-means introduced in A Dundar, J Jin, E Culurciello
% "Convolutional Clustering for Unsupervised Learning"
%
% W - matrix n x m representing m patches (windows) of length n
% D - matrix n x k representing the learned dictionary of k n-dimensional vectors
% max_iter - the maximum number of convolutional k-means iterations
% batch_size - the number of patches for convolution (for speed up)
% wS - window size
% wS_ratio [default = 2] - the ratio of window size (wS) to filter size (s)

if (~exist('max_iter','var'))
    max_iter = 50;
end
[n,m] = size(W);
s = wS(1)/wS_ratio;
c = n/wS(1)^2;
% wS_ratio*s = window size (by default, two times larger than filter size)
filter_size = [s, s, c]; % size of learned filters (centroids)
W_xy = zeros(prod(filter_size), m, 'single'); % patches corresponding to the biggest activations 

% initialize random normalized centroids
% w_i = randn(k, prod(filter_size), 'single'); 
w_i = reshape(W(:,randperm(m,k)),[wS_ratio*s,wS_ratio*s,c,k]);
w_i = reshape(w_i(round(0.25*s):round(1.25*s)-1,round(0.25*s):round(1.25*s)-1,:,:), prod(filter_size), k)';
D = feature_scaling(w_i, 'l2')'; % initialize random normalized centroids

vis = exist('vis','var') && vis && (c == 1 || c == 3); % visualize filters during learning
if (vis), figure; end

for it=1:max_iter
    S = zeros(k,m,'single'); % sparse matrix k x m of distances between all patches and most similar centroids
    for i=1:batch_size:m
        n_batch = min(size(W,2),i+batch_size-1) - i + 1;
        w_i = reshape(W(:,i:i+n_batch-1),[wS_ratio*s,wS_ratio*s,c,n_batch]);
        d = gather(vl_nnconv(gpuArray(w_i),gpuArray(reshape(D,[filter_size,k])),[])); % convolve with all centroids
        % get coordinates of the biggest activation and the most similar centroid
        d_max = max(max(max(d,[],1),[],2),[],3); % the biggest activation for this sample
        sz = size(d);
        for j=1:n_batch
            [y, x, k_max] = ind2sub(sz(1:3),find(d_max(j) == d(:,:,:,j),1,'first'));
            W_xy(:,j+i-1) = reshape(w_i(y:y+s-1,x:x+s-1,:,j),[prod(filter_size),1]);
            S(k_max, j+i-1) = d_max(j);
        end
    end
    D = W_xy*S'+D;
    D = feature_scaling(D','l2')';
    if (it > 1)
        delta = norm(D(:)-D_tmp(:));
        fprintf('it = %d/%d, delta = %5.3f \n', it, max_iter, delta)
        if (delta < 0.1)
            fprintf('centroids do not update, terminated \n')
            break;
        end
    else
        fprintf('it = %d/%d \n', it, max_iter)
    end
    D_tmp = D;
    if (vis)
        imsetshow(reshape(D,[filter_size, k])); drawnow;
    end
end

[~,ids] = max(S,[],1); % which clusters assigned to which windows

end