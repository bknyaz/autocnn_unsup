function [vectors,stats] = feature_scaling(vectors, featScaling, varargin)
n_samples = size(vectors,1);
if (n_samples > 1024)
    batch_size = 1024;
    stats = [];
    n_batches = ceil(n_samples/batch_size);
    for batch_id = 1:n_batches
        samples_ids = (batch_id-1)*batch_size+1:batch_id*batch_size;
        samples_ids(samples_ids > n_samples) = [];
        [vectors(samples_ids,:),stats] = feature_scaling_batch(vectors(samples_ids,:), featScaling, varargin{:});
    end
else
    [vectors,stats] = feature_scaling_batch(vectors, featScaling, varargin{:});
end

end

function [vectors,stats] = feature_scaling_batch(vectors, featScaling, varargin)
% rows are observations (samples), cols are variables

stats = [];

switch (lower(featScaling))
    case 'gray'
        limits = double([min(vectors,[],2) max(vectors,[],2)]); % limits for each observation
        idx = limits(:,2) ~= limits(:,1);
        delta = 1 ./ (limits(idx,2) -  limits(idx,1));
        vectors(idx,:) = bsxfun(@plus, bsxfun(@times, vectors(idx,:), delta), -limits(idx,1).*delta);
        vectors = max(0,min(vectors,1));
    case 'log'
        idx = abs(vectors) <= 1;
        vectors(idx) = 0;
        vectors(~idx) = sign(vectors(~idx)).*log(abs(vectors(~idx)));
    case 'stat'
        if (nargin == 3)
            [vectors,stats.mn,stats.sd] = standardizing(vectors,false,varargin{1});
        elseif (nargin > 3)
            vectors = bsxfun(@minus,vectors,varargin{1});
            varargin{2}(varargin{2}==0) = 1e-10;
            vectors = bsxfun(@rdivide,vectors,varargin{2});
        else
            [vectors,stats.mn,stats.sd] = standardizing(vectors,false,2);
        end
    case 'stat_root'
        vectors = sign(vectors).*(abs(vectors).^0.5);
        [vectors,stats] = feature_scaling(vectors, 'stat', varargin{:});
    case 'l2'
        vectors(sum(vectors,2) == 0,:) = eps;
        stats.norms = arrayfun(@(idx) norm(vectors(idx,:)), (1:size(vectors,1))');
        vectors = bsxfun(@rdivide, vectors, stats.norms);
    case 'l1'
        b = sum(abs(vectors),2);
        vectors(b == 0,:) = eps;
        vectors = bsxfun(@rdivide, vectors, b);
    case 'rootsift'
        vectors = feature_scaling(vectors, 'l1');
        if (nargin == 3)
            feat_scal_pow = varargin{1};
        else
            feat_scal_pow = 0.5;
        end
        vectors = sign(vectors).*(abs(vectors).^feat_scal_pow);
    otherwise
        error('not supported feature scaling method')
end

end

function [I,mn,sd] = standardizing(I, issparse, dim)
good = isfinite(I); 
if (issparse)
    er = 0;
    good = (abs(I) > er) & good;
    mn = mean(I(good));
    I(good) = I(good) - mn;
    sd = std(I(good));
    if (sd > 0)
        I(good) = I(good)/sd;
    end
else
    if (isempty(dim))
        mn = mean(I(:));
        I = bsxfun(@minus,I,mn);
        sd = std(I(:));
        sd(sd==0) = 1e-10;
        I = bsxfun(@rdivide,I,sd);
    else
        mn = mean(I,dim);
        I = bsxfun(@minus,I,mn);
        sd = std(I,0,dim);
        sd(sd==0) = 1e-10;
        I = bsxfun(@rdivide,I,sd);
    end
end
idx = ~isfinite(I);
if (nnz(idx))
    I(idx) = 0;
end
end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            