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