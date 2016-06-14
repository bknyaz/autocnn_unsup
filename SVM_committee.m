function [acc,scores,predicted_labels] = SVM_committee(train_data, test_data, train_labels, test_labels, opts)
% Trains a committee of J>0 SVM models
% opts - model and SVM parameters
% To train SVMs on a GPU GTSVM must be installed: http://ttic.uchicago.edu/~cotter/projects/gtsvm/

J = length(opts.PCA_dim); % the number of SVM models in the committee
acc = zeros(2,J); % predicting accuracies in %
scores = cell(1,J); % SVM scores
predicted_labels = cell(1,J); % test data labels predicted with SVMs
C = 16; % the SVM regularization constant is fixed in all experiments
kernel = 2; % RBF SVM

% check labels
train_labels = double(train_labels(1:size(train_data,1)));
test_labels = double(test_labels(1:size(test_data,1)));
% make sure labels are from 0 to n_classes
train_labels = train_labels-min(train_labels); 
test_labels = test_labels-min(test_labels);
labels = unique(train_labels)';
n_classes = length(labels);
if (n_classes ~= length(unique(test_labels)))
    error('all classes must present in both training and test data')
end
fprintf('%d training labels: \t %s \n', n_classes, num2str(labels)) 
fprintf('%d test labels: \t %s \n', n_classes, num2str(unique(test_labels)'))

if (opts.gpu_svm)
    context = gtsvm;
    proc = 'gpu';
else
    proc = 'cpu';
    % For LIBSVM to make scores label-consistent for arbitrary sets of training labels
    % it is necessary to sort training samples so that the first training labels are 0:n_classes-1
    id_first = zeros(n_classes,1);
    for i=1:n_classes, id_first(i) = find(train_labels == labels(i),1,'first'); end
    id_rest = ~ismember(1:length(train_labels),id_first);
    train_labels = cat(1,train_labels(id_first),train_labels(id_rest));
    train_data = cat(1,train_data(id_first,:),train_data(id_rest,:));
    % convert to double
    train_data = double(train_data);
    test_data = double(test_data);
end

time_train = 0;
time_test = 0;
% train a committee
for j=1:J
    p_j = opts.PCA_dim(j);
    tic;
    fprintf('%d/%d, SVM model for PCA dim (p_j) = %d \n', j, J, p_j)
    fprintf('- using a %s: training...', upper(proc))
    % normalize features
    train_data_dim = feature_scaling(train_data(:,1:p_j), opts.norm);
    if (opts.gpu_svm)
        context.initialize(train_data_dim, train_labels, true, C, 'gaussian', 1/(size(train_data_dim,2)), 0, 0, false);
        context.optimize( 0.01, 1000000 );
    else
        model = svmtrain(train_labels, train_data_dim, sprintf('-t %d -q -c %f', kernel, C));
    end
    time_train = time_train+toc;
    tic;
    fprintf('predicting...')
    test_data_dim = feature_scaling(test_data(:,1:p_j), opts.norm);
    if (opts.gpu_svm)
        scores{j} = context.classify(test_data_dim);
        [~,predicted_labels{j}] = max(scores{j},[],2);
        predicted_labels{j} = predicted_labels{j}-min(predicted_labels{j});
        acc(1,j) = (1-nnz(predicted_labels{j} ~= test_labels)/length(predicted_labels{j}))*100;
        fprintf('\n')
    else
        [predicted_labels{j}, accuracy, scores{j}] = svmpredict(test_labels, test_data_dim, model);
        acc(1,j) = accuracy(1);
    end
    scores{j} = single(scores{j});
    predicted_labels{j} = single(predicted_labels{j});
    % predict labels using the committee of SVM scores (average or sum scores and take maximum)
    [~, ~, acc(2,j)] = predict_labels(mean(cat(3,scores{1:j}),3), labels, test_labels, n_classes, ~opts.gpu_svm);
    time_test = time_test+toc;
    
    fprintf('- Accuracy of a single SVM model = %f \n', acc(1,j))
    fprintf('- Accuracy of a committee of %d SVM model(s)  = %f \n', j, acc(2,j));
end

% clean GTSVM resources
if (opts.gpu_svm)
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

function [predicted_labels, predicted_labels_second, acc] = predict_labels(scores, train_labels, test_labels, n_classes, onevsone)

u = unique(train_labels,'stable');
predicted_labels_second = []; % second guesses
if (onevsone)
    predicted_labels = zeros(length(test_labels),1);
    predicted_labels_second = predicted_labels;
    for k=1:length(test_labels)
        vote = zeros(1,n_classes);
        for model_id=1:size(scores,3)
            p = 1;
            for i=1:n_classes
                for j=i+1:n_classes

                    dec_value = scores(k,p,model_id);

                    if(dec_value > 0.0)
                        vote(i) = vote(i) + 1;
                    else
                        vote(j) = vote(j) + 1;
                    end
                    p = p + 1;
                end
            end
        end
        vote_max_idx = find(vote == max(vote));
        vote_max_idx = vote_max_idx(1);
        predicted_labels(k) = u(vote_max_idx);
        m = vote(vote_max_idx);
        vote_max_idx = 1;
        for i=1:n_classes
            if(vote(i) > vote(vote_max_idx) && vote(i) < m)
                vote_max_idx = i;
            end
        end
        predicted_labels_second(k) = u(vote_max_idx);
    end
else
    [~,idx] = max(scores,[],2);
    predicted_labels = idx-1;
end
acc = nnz(predicted_labels == test_labels)/numel(test_labels)*100;
end