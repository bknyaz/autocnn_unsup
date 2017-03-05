function [acc,scores,predicted_labels,svm_params,model] = classifier_committee(train_data, test_data, train_labels, test_labels, opts)
% Trains a committee of J>0 SVM (or LDA) models
% opts - model and SVM (or LDA) parameters
% To train SVMs on a GPU GTSVM must be installed: http://ttic.uchicago.edu/~cotter/projects/gtsvm/

J = max(1,length(opts.PCA_dim)); % the number of SVM (or LDA) models in the committee
acc = zeros(2,J); % predicting accuracies in %
scores = cell(1,J); % SVM (or LDA) scores
predicted_labels = cell(1,J); % test data labels predicted with SVMs (or LDA)
model = cell(1,J);

% the SVM regularization constant for RBF SVM (is fixed in all experiments)
if (~isfield(opts,'SVM_C')), C = 16; else C = opts.SVM_C; end
kernel = 2; % RBF SVM if libsvm
svm_params = [];
% check labels
train_labels = double(train_labels);
test_labels = double(test_labels);
% make sure that labels are in the range [0,n_classes]
train_labels = train_labels-min(train_labels); 
test_labels = test_labels-min(test_labels);
labels = unique(train_labels)';
n_classes = length(labels);
if (n_classes ~= length(unique(test_labels)))
    warning('all classes must present in both training and test data')
end
fprintf('%d training labels: \t %s \n', n_classes, num2str(labels)) 
fprintf('%d test labels: \t %s \n', n_classes, num2str(unique(test_labels)'))

cv_mode = false;
if (strcmpi(opts.classifier,'gtsvm'))
    context = gtsvm;
    proc = 'gpu';
else
    proc = 'cpu';
    % For LIBSVM to make scores ordered according to labels (0-n_classes) for arbitrary sets of training labels
    % it is necessary to sort training samples so that the first training labels are 0:n_classes-1
    id_first = zeros(n_classes,1);
    for i=1:n_classes, id_first(i) = find(train_labels == labels(i),1,'first'); end
    id_rest = ~ismember(1:length(train_labels),id_first);
    train_labels = cat(1,train_labels(id_first),train_labels(id_rest));
    train_data = cat(1,train_data(id_first,:),train_data(id_rest,:));
    % convert to double
    train_data = double(train_data);
    
    test_labels_tmp = test_labels;
    n = size(test_data,1)/length(test_labels);
    if (size(test_data,1) ~= length(test_labels))
        if (mod(size(test_data,1),length(test_labels)) == 0 && n > 1)
            test_labels_tmp = repmat(test_labels,n,1); % in case of simple data augmentation (flip)
        else
            error('not supported mode')
        end
    end
    if (strcmpi(opts.classifier,'liblinear'))
        if (~isfield(opts,'SVM_C') || ~isfield(opts,'SVM_B'))
            cv_mode = true;
            if isfield(opts,'dataset') && strcmpi(opts.dataset,'mnist')
                [C_val,B_val] = meshgrid([1e-4,2e-4,4e-4,8e-4,16e-4,32e-4],[0,3,5])
            else
                [C_val,B_val] = meshgrid([1e-4,2e-4,4e-4,8e-4],[0,3,5])
            end
        else
            C = opts.SVM_C;
            B = opts.SVM_B;
        end
    end
end

time_train = 0;
time_test = 0;
% train a committee
for j=1:J
    if (isempty(opts.PCA_dim) || opts.PCA_dim(j) == 0)
      p_j = size(train_data,2);
    else
      p_j = opts.PCA_dim(j);
    end
    tic;
    fprintf('%d/%d, SVM model for PCA dim (p_j) = %d \n', j, J, p_j)
    fprintf('- using a %s: training...', upper(proc))
    if (J==1)
        train_data_dim = train_data;
        clear train_data;
    else
        if (issparse(train_data) && p_j == size(train_data,2))
            train_data_dim = train_data;
        else
            train_data_dim = train_data(:,1:p_j);
        end
    end
    % normalize features
    if (~isempty(opts.norm) && ~issparse(train_data_dim))
      train_data_dim = feature_scaling(train_data_dim, opts.norm);
    end
    if (strcmpi(opts.classifier,'gtsvm'))
        context.initialize(train_data_dim, train_labels, true, C, 'gaussian', 1/(size(train_data_dim,2)), 0, 0, false);
        context.optimize( 0.01, 1000000 );
    elseif (strcmpi(opts.classifier,'libsvm'))
        model{j} = svmtrain(train_labels, train_data_dim, sprintf('-t %d -q -c %f', kernel, C));
        predict_fn = @svmpredict;
    elseif (strcmpi(opts.classifier,'liblinear'))
        if (~issparse(train_data_dim) && ~cv_mode)
            train_data_dim = sparse(train_data_dim);
        end
        if (cv_mode)
            n_max = 10^4;
            fprintf('cross-validation of C and B ... \n')
            if (size(test_data,1) > length(test_labels))
                fprintf('using the first %d samples for validation ... \n', opts.n_train)
                cv_ids = 1:min(n_max,opts.n_train);
                train_data_dim_cv = sparse(train_data_dim(cv_ids,:));
            else
                if (length(train_labels) < n_max)
                    cv_ids = 1:length(train_labels);
                    train_data_dim_cv = sparse(train_data_dim);
                else
                    cv_ids = 1:min(n_max,length(train_labels));
                    train_data_dim_cv = sparse(train_data_dim(cv_ids,:));
                end
            end
            acc_cv = [];
            for k=1:numel(C_val)
                fprintf('%d/%d, C=%f,B=%f \n', k, numel(C_val), C_val(k), B_val(k))
                acc_cv(k) = train(train_labels(cv_ids), train_data_dim_cv, sprintf('-v 5 -s 1 -q -c %f -B %f', C_val(k), B_val(k)));
            end
            [~,k] = max(acc_cv);
            C = C_val(k(1));
            B = B_val(k(1));
            fprintf('best C = %f and B = %f \n', C, B)
            if (size(test_data,1) > length(test_labels))
                clear train_data_dim_cv
                train_data_dim = sparse(train_data_dim);
            else
                if (length(train_labels) < n_max)
                    train_data_dim = train_data_dim_cv;
                    clear train_data_dim_cv
                else
                    clear train_data_dim_cv
                    train_data_dim = sparse(train_data_dim);
                end
            end
        end
        model{j} = train(train_labels, train_data_dim, sprintf('-s 1 -q -c %f -B %f', C, B));
        predict_fn = @predict;
    elseif (strcmpi(opts.classifier,'lda'))
        model{j} = fitcdiscr(train_data_dim, train_labels, 'SaveMemory','on');
    else
        error('not supported classifier')
    end
    time_train = time_train+toc;
    if (J==1), clear train_data_dim; end
    tic;
    fprintf('predicting...')
    if (J==1) 
        test_data_dim = test_data;
        clear test_data;
    else
        if (issparse(test_data) && p_j == size(test_data,2))
            test_data_dim = test_data;
        else
            test_data_dim = test_data(:,1:p_j);
        end
    end
    if (~isempty(opts.norm) && ~issparse(test_data_dim))
      test_data_dim = feature_scaling(test_data_dim, opts.norm);
    end
    if (strcmpi(opts.classifier,'libsvm') || strcmpi(opts.classifier,'liblinear'))
       [scores{j}, predicted_labels{j}, acc(1,j)] = predict_batches(test_data_dim, test_labels_tmp, test_labels, labels, model{j}, predict_fn, opts);
    else
        if (strcmpi(opts.classifier,'gtsvm'))
            scores{j} = context.classify(test_data_dim);
        else
            [~, scores{j}] = predict(model{j}, test_data_dim);
        end
        n = size(scores{j},1)/length(test_labels);
        if (mod(size(scores{j},1),length(test_labels)) == 0 && n > 1)
            scores{j} = squeeze(mean(reshape(scores{j},length(test_labels),n,size(scores{j},2)),2));
        end
        [~,predicted_labels{j}] = max(scores{j},[],2);
        predicted_labels{j} = predicted_labels{j}-min(predicted_labels{j});
        acc(1,j) = nnz(predicted_labels{j} == test_labels)/length(predicted_labels{j})*100;
        fprintf('\n')
    end
    scores{j} = single(scores{j});
    predicted_labels{j} = single(predicted_labels{j});
    % predict labels using the committee of SVM scores (average or sum scores and take maximum)
    [predicted_labels_c, ~, acc(2,j)] = predict_labels(mean(cat(3,scores{1:j}),3), labels, test_labels, n_classes, strcmpi(opts.classifier,'libsvm'));
    time_test = time_test+toc;
    
    fprintf('- Accuracy of a single classifier model = %f (%d/%d)\n', acc(1,j), nnz(predicted_labels{j} == test_labels), length(test_labels))
    fprintf('- Accuracy of a committee of %d classifier model(s)  = %f (%d/%d)\n', j, acc(2,j), nnz(predicted_labels_c == test_labels), length(test_labels));
end

% clean GTSVM resources
if (strcmpi(opts.classifier,'gtsvm'))
    try
        context.deinitialize;
        context.deinitialize_device;
    catch e
        warning(e.message)
    end
end

fprintf('%s \n', upper('timing data'))

fprintf('training: \t %3.2f sec (committee) \t\t\t %3.2f sec (avg single) \n \t\t %3.2f samples/sec (committee) \t %3.2f samples/sec (avg single) \n', ...
    time_train, time_train/J, length(train_labels)/time_train, length(train_labels)/time_train*J)

fprintf('prediction: \t %3.2f sec (committee) \t\t\t %3.2f sec (avg single) \n \t\t %3.2f samples/sec (committee) \t %3.2f samples/sec (avg single) \n', ...
    time_test, time_test/J, length(test_labels)/time_test, length(test_labels)/time_test*J)
end