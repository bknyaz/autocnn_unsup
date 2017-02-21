function X_n = autoconv_recursive_2d(X, conv_order_MAX, filter_size, norm_type)
% X - a random image (patch) or batches in the spatial domain
% n_MAX - the last autoconvolution order
% filter_size - desired size of returned patches
% X_n - a collection of autoconvolutional patches of orders n=0,1,...,n_MAX
% norm_type: 1 for layer 1 filters and any other value for other layers
% Patches are normalized in the range [0,1]

samplingRatio = 2;
X_n = cell(1,conv_order_MAX+1);
for conv_order = 0:conv_order_MAX
    if (conv_order > 0)
      X = real(autoconv_2d(X, norm_type));
      if (rand > 0.5)
          X_sampled = imresize(X, 1/samplingRatio, 'bicubic'); 
      else
          X_sampled = downsample(X, samplingRatio, 'space');
      end
      if (conv_order > 1)
        X = X_sampled;
      end
      if (size(X_sampled,1) ~= filter_size(1))
        k = floor(filter_size(1)/size(X_sampled,1)*100)/100;
        X_sampled = imresize(X_sampled,k);
      end
      X_n{1,conv_order+1} = single(X_sampled);
    else
        if (size(X,1) ~= filter_size(1))
            X_sampled = imresize(X, filter_size(1)/size(X,1));
            if (conv_order_MAX == 0)
                X = X_sampled;
            end
        else
            X_sampled = X;
        end
        X_n{1,conv_order+1} = X_sampled;
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
    X = bsxfun(@rdivide,X,sd+1e-10);
else
    m = mean(mean(mean(X,1),2),3);
    X = bsxfun(@minus,X,m);
    sd = std(std(std(X,0,1),0,2),0,3);
    X = bsxfun(@rdivide,X,sd+1e-10);
end
sz = size(X);
X = padarray(X, sz(1:2)-1, 'post'); % zero-padding to compute linear convolution
X = ifft2(fft2(X).^2);
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