function [scores, predicted_labels, acc] = predict_batches(test_data_dim, test_labels_tmp, test_labels, labels, model, predict_fn, opts)

n_samples = size(test_data_dim,1);
if (n_samples > 1024)
    batch_size = 1024;
    n_batches = ceil(n_samples/batch_size);
    for batch_id = 1:n_batches
        samples_ids = (batch_id-1)*batch_size+1:batch_id*batch_size;
        samples_ids(samples_ids > n_samples) = [];
        [scores(samples_ids,:), predicted_labels(samples_ids)] = predict_batch(test_data_dim(samples_ids,:), test_labels_tmp(samples_ids), model, predict_fn, opts);
    end
    [predicted_labels, ~, accuracy] = predict_labels(scores, labels, test_labels_tmp, length(labels), strcmpi(opts.classifier,'libsvm'));
else
    [scores, predicted_labels, accuracy] = predict_batch(test_data_dim, test_labels_tmp, model, predict_fn, opts);
end

n = length(predicted_labels)/length(test_labels);
if (mod(length(predicted_labels),length(test_labels)) == 0 && n > 1)
    scores = squeeze(mean(reshape(scores,length(test_labels),n,size(scores,2)),2));
    [predicted_labels, ~, accuracy] = predict_labels(scores, labels, test_labels, length(labels), strcmpi(opts.classifier,'libsvm'));
end
acc = accuracy(1);

end

function [scores, predicted_labels, acc] = predict_batch(test_data_dim, test_labels_tmp, model, predict_fn, opts)
if (strcmpi(opts.classifier,'liblinear') && ~issparse(test_data_dim))
  test_data_dim = sparse(double(test_data_dim));
elseif (~issparse(test_data_dim))
  test_data_dim = double(test_data_dim);
end
[predicted_labels, acc, scores] = predict_fn(test_labels_tmp, test_data_dim, model);
end