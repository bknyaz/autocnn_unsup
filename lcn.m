function [x_ijk, c] = lcn(x_ijk, c, is_vl, sigma_coef)
% Local contrast normalization
% according to Jarrett et al. "What is the Best Multi-Stage Architecture for Object Recognition?"
% i - feature map index
% j,k and p,q - spatial indices
% c - constant (mean(sigma_jk(:)))
% is_vl - true to use MatConvNet
% in our implementation, x_ijk - a batch of images

is_vl = ~exist('is_vl','var') || is_vl;
gpu = isa(x_ijk,'gpuArray');
sz = size(x_ijk);
if (length(sz) < 3), sz(3) = 1; end
if (length(sz) < 4), sz(4) = 1; end

sigma = sz(1)/16; % Gaussian window sigma (decrease the value to increase speed)

if (mod(round(sigma*4+1),2) == 0)
  sigma = sigma+0.25; % to make size of w_pq non even
end
% initialize a Gaussian weighting window
w_pq = repmat(single(fspecial('gaus', round(sigma*4+1), sigma*sigma_coef)), [1,1,sz(3)]); % sum(w(:)) = sz(3)
w_pq = w_pq./sz(3); % so that sum for all windows = 1, sum_ipq(w) = 1
if (gpu), w_pq = gpuArray(w_pq); end

% Subtractive normalization
if (is_vl)
    x_ijk = bsxfun(@minus, x_ijk, vl_nnconv(x_ijk, w_pq, [], 'pad', round(sigma*2))); % v_ijk 
else
    w_pq = padarray(w_pq, sz(1:2)-1,0,'post');
    x_ijk = padarray(x_ijk, repmat(round(sigma*4+1),1,2)-1,0,'post');
    for d=1:2, w_pq = fft(w_pq,[],d);  end
    x_ijk = bsxfun(@minus, x_ijk, conv_fft(fft2(x_ijk), w_pq, sz, false)); % v_ijk 
end

% Divisive normalization
% weighted standard deviation of all features over a spatial neighborhood
if (is_vl)
    sigma_jk = sqrt(vl_nnconv(x_ijk.^2, w_pq, [], 'pad', round(sigma*2)));
else
    sigma_jk = sqrt(conv_fft(fft2(x_ijk.^2), w_pq, sz, true));
    offset = floor(([size(x_ijk,1),size(x_ijk,2)] - sz(1:2))./2);
    x_ijk = x_ijk(offset+1:end-offset,offset+1:end-offset,:,:);
end

if (~exist('c','var') || isempty(c))
    c = mean(sigma_jk(:)); % mean over all feature maps and images in a batch
end
x_ijk = bsxfun(@rdivide, x_ijk, max(c,sigma_jk)); % y_ijk

if (gpu), c = gather(c); end

end

% Convolution wrapper for convenience
function fmaps = conv_fft(fmaps, filters, sample_size, keep_size)
% using Matlab in the frequency domain
fmaps = squeeze(bsxfun(@times, permute(fmaps,[1:3,5,4]), filters));
for d=1:2, fmaps = ifft(fmaps,[],d); end
fmaps = real(sum(fmaps,3));
if (keep_size)
    sz = size(fmaps);
    offset = floor((sz(1:2) - sample_size(1:2))./2);
    fmaps = fmaps(offset+1:end-offset,offset+1:end-offset,:,:);
end
    
end