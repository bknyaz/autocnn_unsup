function [data, PCA_matrix, data_mean, L_regul] = pca_zca_whiten(data, opts, varargin)
% Performs data whitening and PCA or ZCA
% 
% varargin - PCA_matrix, data_mean, L_regul
% opts - a structure with parameters:
% pca_mode - 'pcawhiten', 'zcawhiten', 'pca' or 'zca'
% other parameter are optional:
% [pca_dim] - the number of PCA dimensionalities (principal components) to be used for transformation 
% [pca_fraction] (ignored if pca_dim is specified) - the fraction (from 0 to 1) of data variance to be preserved 
% [pca_fast] - randomized (Fast Singular Value Decomposition by Halko N. et al.) or fixed (eigenvectors based) PCA 
% [pca_max_dims] (ignored if pca_fast is specified) - the threshold to decide if randomized or fixed PCA is performed
% [pca_epsilon] - regularization constant for whitening
% [vis] - visualize the covariance matrix after transformation

if (nargin < 2)
    error('not enough input arguments')
elseif (nargin > 2)
    projection = true;
    PCA_matrix = varargin{1};
    data_mean = varargin{2};
    if (nargin > 4)
        L_regul = varargin{3};
    else
        L_regul = [];
    end
    dim = size(PCA_matrix,2);
    if (size(data,2) ~= size(PCA_matrix,1))
        error('invalid data')
    end
else
    projection = false;
    if (size(data,1) <= 1)
      error('data are invalid')
    end
    if (isfield(opts,'pca_dim') && ~isempty(opts.pca_dim))
        dim = opts.pca_dim; % the number of PCA dimensionalities (principal components)
        dim_str = num2str(dim);
        if (size(data,2) <= dim && isempty(strfind(opts.pca_mode,'zca')))
          error('the number of samples must be greater than the number of desired dimensions')
        end
    elseif (isfield(opts,'pca_fraction'))
        pca_fraction = opts.pca_fraction;
        dim = [];
        dim_str = 'pca_dim';
    else
        pca_fraction = 0.99; % preserve 99% of data variance
        dim = [];
        dim_str = 'pca_dim';
    end
    if (strfind(opts.pca_mode,'zca'))
        dim_str = num2str(size(data,2));
    end
end
if (~isfield(opts,'verbose'))
    opts.verbose = true;
end
[m,n] = size(data);
if (projection)
    if (opts.verbose), fprintf('-> %s: projection %dx%d -> %dx%d \n', opts.pca_mode, m, n, m, dim); end
    data = bsxfun(@minus, data, data_mean);
else
    if (~isfield(opts,'pca_fast'))
        if (~isfield(opts,'pca_max_dims'))
            opts.pca_max_dims = 8*10^3;
        end
        if (n > opts.pca_max_dims)
            warning('PCA switched to the fast mode')
            opts.pca_fast = true;
        else
            opts.pca_fast = false;
        end
    end
    str = '';
    if (opts.pca_fast)
        str = ' (randomized)';
    end
    if (opts.verbose), fprintf('-> performing%s %s: %dx%d -> %dx%s \n', str, opts.pca_mode, m, n, m, dim_str); end
    
    data_mean = mean(data);
    data = bsxfun(@minus, data, data_mean);
    
    if (isempty(dim))
        dim = n;
        select_dim = true;
    else
        select_dim = false;
        dim = min(dim, size(data,2));
    end
    if (opts.pca_fast)
        [evectors,evalues,~] = fsvd(data', dim, 2);
        evalues = evalues/sqrt(m-1);
        evalues = diag(evalues);
    else
        if (~strcmpi(class(data),'double'))
            data = single(data);
        end
        cov_data = (data' * data) / (m-1);
        if (~strcmpi(class(cov_data),'double'))
            cov_data = double(cov_data); % required for eigs
        end
        if (dim < size(cov_data,1))
            [evectors, evalues] = eigs(cov_data,dim);
            evalues = diag(evalues);
        else
            [evectors, evalues] = eig(cov_data);
            evalues = diag(evalues);
            [evalues,id] = sort(evalues,'descend');
            evectors = evectors(:,id);
        end
        clear cov_data;
        evalues = sqrt(single(evalues));
        evectors = single(evectors);
    end
    if (select_dim)
        var_fraction = cumsum(evalues)./sum(evalues(:));
        dim = find(var_fraction > pca_fraction, 1, 'first');
        if (isempty(dim))
            dim = n;
        end
        if (opts.verbose)
            fprintf('pca_dim (the number of PCA dimensionalities, %3.2f %% of data variance is preserved): %d \n', ...
                pca_fraction*100, dim)
        end
    end
    PCA_matrix = evectors(:,1:dim);
    clear evectors;
    if (strfind(lower(opts.pca_mode),'whiten'))
        if (~isfield(opts,'pca_epsilon') || isempty(opts.pca_epsilon))
            opts.pca_epsilon = 1e-5;
        end
        if (opts.verbose), fprintf('whitening with regul = %1.3e \n', opts.pca_epsilon); end
        L_regul = diag(1./(evalues(1:dim)+opts.pca_epsilon));
    end
        
    % adopted from Matlab pcacov.m: "Enforce a sign convention on the coefficients -- the largest element in each column will have a positive sign."
    [p,d] = size(PCA_matrix);
    [~,maxind] = max(abs(PCA_matrix),[],1);
    colsign = sign(PCA_matrix(maxind + (0:p:(d-1)*p)));
    PCA_matrix = bsxfun(@times,PCA_matrix,colsign);
end

data = data*PCA_matrix;
if (~isempty(strfind(lower(opts.pca_mode),'whiten')) && ~isempty(L_regul))
    data = data*L_regul;
end
if (strfind(opts.pca_mode,'zca'))
    data = data*PCA_matrix';
end

% check how well new data are decorrelated
if (isfield(opts,'vis') && opts.vis)
    figure,imagesc(cov(data))
end
end


% This code adopted from other guys
function [U,S,V] = fsvd(A, k, i, usePowerMethod, treatNan)
% FSVD Fast Singular Value Decomposition 
% 
%   [U,S,V] = FSVD(A,k,i,usePowerMethod) computes the truncated singular
%   value decomposition of the input matrix A upto rank k using i levels of
%   Krylov method as given in [1], p. 3.
% 
%   If usePowerMethod is given as true, then only exponent i is used (i.e.
%   as power method). See [2] p.9, Randomized PCA algorithm for details.
% 
%   [1] Halko, N., Martinsson, P. G., Shkolnisky, Y., & Tygert, M. (2010).
%   An algorithm for the principal component analysis of large data sets.
%   Arxiv preprint arXiv:1007.5510, 0526. Retrieved April 1, 2011, from
%   http://arxiv.org/abs/1007.5510. 
%   
%   [2] Halko, N., Martinsson, P. G., & Tropp, J. A. (2009). Finding
%   structure with randomness: Probabilistic algorithms for constructing
%   approximate matrix decompositions. Arxiv preprint arXiv:0909.4061.
%   Retrieved April 1, 2011, from http://arxiv.org/abs/0909.4061.
% 
%   See also SVD.
% 
%   Copyright 2011 Ismail Ari, http://ismailari.com.

    if nargin < 3
        i = 1;
    end

    % Take (conjugate) transpose if necessary. It makes H smaller thus
    % leading the computations to be opts.pca_faster
    if size(A,1) < size(A,2)
        A = A';
        isTransposed = true;
    else
        isTransposed = false;
    end

    n = size(A,2);
    l = k + 2;

    % Form a real n?l matrix G whose entries are iid Gaussian r.v.s of zero
    % mean and unit variance
    G = randn(n,l);


    if nargin >= 4 && usePowerMethod
        % Use only the given exponent
        H = A*G;
        for j = 2:i+1
            H = A * (A'*H);
        end
    else
        % Compute the m?l matrices H^{(0)}, ..., H^{(i)}
        % Note that this is done implicitly in each iteration below.
        H = cell(1,i+1);
        H{1} = A*G;
        for j = 2:i+1
            H{j} = A * (A'*H{j-1});
        end

        % Form the m?((i+1)l) matrix H
        H = cell2mat(H);
    end

    % Using the pivoted QR-decomposiion, form a real m?((i+1)l) matrix Q
    % whose columns are orthonormal, s.t. there exists a real
    % ((i+1)l)?((i+1)l) matrix R for which H = QR.  
    % XXX: Buradaki column pivoting ile yap?lmayan hali.
    [Q,~] = qr(H,0);

    % Compute the n?((i+1)l) product matrix T = A^T Q
    T = A'*Q;

    % Form an SVD of T
    if nargin >= 5 && treatNan
        idx_T = ~isfinite(T);
        if (nnz(idx_T))
            T(idx_T) = 0;
            warning('\n %d NaN values replaced with 0s in SVD ', nnz(idx_T))
        end
    end
    [Vt, St, W] = svd(T,'econ');

    % Compute the m?((i+1)l) product matrix
    Ut = Q*W;

    % Retrieve the leftmost m?k block U of Ut, the leftmost n?k block V of
    % Vt, and the leftmost uppermost k?k block S of St. The product U S V^T
    % then approxiamtes A. 

    if isTransposed
        V = Ut(:,1:k);
        U = Vt(:,1:k);     
    else
        U = Ut(:,1:k);
        V = Vt(:,1:k);
    end
    S = St(1:k,1:k);
end