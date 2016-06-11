function [x_ijk,  c] = lcn(x_ijk, c)
% Local contrast normalization
% according to Jarrett et al. "What is the Best Multi-Stage Architecture for Object Recognition?"
% i - feature map index
% j,k and p,q - spatial indices
% c - constant (mean(sigma_jk(:)))
% in our implementation, x_ijk - a batch of images

gpu = strcmpi(class(x_ijk),'gpuArray');
sz = size(x_ijk);
if (length(sz) < 3), sz(3) = 1; end
if (length(sz) < 4), sz(4) = 1; end

sigma = ceil(sz(1)/16); % Gaussian window sigma

% initialize a Gaussian weighting window
w_pq = repmat(single(fspecial('gaus', sigma*4+1, sigma*1.5)), [1,1,sz(3)]); % sum(w(:)) = sz(3)
w_pq = w_pq./sz(3); % so that sum for all windows = 1, sum_ipq(w) = 1
if (gpu), w_pq = gpuArray(w_pq); end

% Subtractive normalization
x_ijk = bsxfun(@minus, x_ijk, vl_nnconv(x_ijk, w_pq, [], 'pad', sigma*2)); % v_ijk 

% Divisive normalization
% weighted standard deviation of all features over a spatial neighborhood
sigma_jk = sqrt(vl_nnconv(x_ijk.^2, w_pq, [], 'pad', sigma*2));
if (~exist('c','var') || isempty(c))
    c = mean(sigma_jk(:)); % mean over all feature maps and images in a batch
end
x_ijk = bsxfun(@rdivide, x_ijk, max(c,sigma_jk)); % y_ijk

if (gpu), c = gather(c); end

end