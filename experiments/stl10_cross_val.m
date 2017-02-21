% Cross-validaton of number of filters, filter size and recursive autoconvolution orders

%% You need to set your own paths to respective dependencies
opts = [];
opts.vlfeat = '/home/boris/Project/3rd_party/vlfeat/toolbox/mex/mexa64';
opts.matconvnet = '/home/boris/Project/3rd_party/matconvnet';
liblinear_path = '/home/boris/Project/3rd_party/liblinear/matlab';
gtsvm_path = '/home/boris/Project/3rd_party/gtsvm/mex'; % for some experiments

% to solve the issue with locating vl_kmeans by Matlab (sometimes it happens)
addpath(opts.vlfeat)
cd /home/boris/Project/3rd_party/vlfeat/bin/glnxa64
vl_kmeans(rand(100,5), 2);
[d,~,~] = fileparts(mfilename('fullpath'));
cd(d)
vl_kmeans(rand(100,5), 2);
addpath(strcat(d,'/../'))

stl10_dir = '/home/boris/Project/autocnn_unsup/data/stl10'; % path to stl10 mat files

%% STL-10
opts.dataDir = stl10_dir;
opts.gtsvm = '';
opts.liblinear = liblinear_path;
opts.n_train = 1e3;
opts.n_folds = 10;
opts.PCA_dim = [];
opts.val = true; % learn new filters for each fold
results = {};
n_filters_list = [96,256,512];
filter_sizes = [9:4:29];
RAs = {'0','0_3'};
file_name = 'stl10_cross-val_autocnn-s1';
n1 = 1;
for n_filters = n_filters_list
    n2 = 1;
    for filter_size = filter_sizes
        n3 = 1;
        for conv_order = {'0','0_3'}
            opts.arch = sprintf('%dc%d-20p-conv%s', n_filters, filter_size, conv_order{1})
            r = autocnn_stl10(opts, 'conv_norm', '')
            results{n1,n2,n3} = {r.acc, r.opts}
            save(strcat(file_name,'.mat'),'results','-v7.3')
            n3 = n3+1;
        end
        n2 = n2+1;
    end
    n1 = n1+1;
end

% load(strcat(file_name,'.mat'))
results_arr = [];
n1 = 1;
for n_filters = n_filters_list
    n2 = 1;
    for filter_size = filter_sizes
        n3 = 1;
        for conv_order = RAs
            acc = [];
            for fold=1:numel(results{n1,n2,n3}{1})
                acc(fold) = results{n1,n2,n3}{1}{fold}(1);
            end
            results_arr(n1,n2,n3) = mean(acc);
            n3 = n3+1;
        end
        n2 = n2+1;
    end
    n1 = n1+1;
end

figure
plot(filter_sizes,results_arr(:,:,1), '-o')
hold on
plot(filter_sizes,results_arr(:,:,2))
% matlab2tikz(strcat(file_name,'.tex'))