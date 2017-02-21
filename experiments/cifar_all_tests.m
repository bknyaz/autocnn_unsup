% Experiments on the CIFAR-10 and CIFAR-100 test sets

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

%% CIFAR-10(400)
opts.cifar100 = false;
opts.gtsvm = '';
opts.liblinear = liblinear_path;
opts.n_train = 4e3;
opts.n_folds = 10;
opts.PCA_dim = [];

%% Learning methods
results = {};
n1 = 1;
for learn_method = {'ica','pca','kmeans'}
    n2 = 1;
    for conv_order = {'0','0_3'}
        if strcmpi(conv_order{1},'0')
            filter_size = 9;
            if strcmpi(learn_method{1},'ica') || strcmpi(learn_method{1},'pca')
                filter_size = 11;
            end
        else
            filter_size = 13;
        end
        opts.arch = sprintf('256c%d-8p-conv%s', filter_size, conv_order{1})
        r = autocnn_cifar(opts, 'learning_method', learn_method{1})
        results{n1,n2} = {r.acc, r.opts}
        save('cifar10_learning_methods_autocnn-s1.mat','results','-v7.3')
        n2 = n2+1;
    end
    n1 = n1+1;
end

%% Shared vs Independent filters
opts.arch = '128c9-3p-conv0_3__32g-4ch-64c7-5p-conv0_2'
autocnn_cifar(opts, 'shared_filters', false)
autocnn_cifar(opts, 'shared_filters', true) % default

%% Comparison to Exemplar-CNN
archs = {[64,128,256],[92,256,512],[256,1024,2048]};
RAs = {{'0_3','0_3','0_2'},{'0','0','0'}}; % with RA and without (Raw)
for arch = archs
    for RA = RAs
        opts.arch = sprintf('%dc5-3p-conv%s__1g-%dch-%dc5-3p-conv%s__1g-%dch-%dc5-3p-conv%s', arch{1}(1),RA{1}{1},arch{1}(1),arch{1}(2),RA{1}{2},arch{1}(2),arch{1}(3),RA{1}{3})
        if (arch{1}(1) == 64)
            autocnn_cifar(opts) % no augmentation
            autocnn_cifar(opts, 'conv_norm', '') % also compute without Rootsift
        end
        autocnn_cifar(opts, 'augment', true)
    end
end
 
%% Comparison to CONV-WTA
opts.n_train = 50e3;
opts.n_folds = 1;
opts.PCA_dim = [];

% 1 layer
opts.arch = '256c9-8p-conv0' % Raw
autocnn_cifar(opts)
opts.arch = '256c13-8p-conv0_3' % RA
autocnn_cifar(opts)

% 2 layers
opts.arch = '256c5-3p-conv0__1g-256ch-1024c5-5p-conv0' % Raw
autocnn_cifar(opts)
opts.arch = '256c5-3p-conv0_3__1g-256ch-1024c5-5p-conv0_2' % RA
autocnn_cifar(opts)

opts.PCA_dim = 4096; % with PCA
opts.arch = '256c5-3p-conv0__1g-256ch-1024c5-5p-conv0' % Raw
autocnn_cifar(opts)
opts.arch = '256c5-3p-conv0_3__1g-256ch-1024c5-5p-conv0_2' % RA
autocnn_cifar(opts)

% 3 layers
opts.PCA_dim = 4096; % without PCA is not feasible in this implementation
opts.arch = '256c5-3p-conv0__1g-256ch-1024c5-3p-conv0__1g-1024ch-2048c5-3p-conv0' % Raw
autocnn_cifar(opts)
autocnn_cifar(opts, 'augment', true)
opts.arch = '256c5-3p-conv0_3__1g-256ch-1024c5-3p-conv0_3__1g-1024ch-2048c5-3p-conv0_2' % RA
autocnn_cifar(opts)
autocnn_cifar(opts, 'augment', true)

%% Final tests with AutoCNN-L32
opts.gtsvm = gtsvm_path;
opts.liblinear = '';
tests = {[4e3,10],[50e3,1]};
for test=tests
    opts.n_train = test{1}(1);
    opts.n_folds = test{1}(2);
    opts.PCA_dim = 1000;
    % No augmentation with RA
    opts.arch = '1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
    autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'})

    % Augmentation
    opts.PCA_dim = 1500;
    opts.arch = '1024c5-3p-conv0__32g-32ch-256c5-3p-conv0__32g-256ch-1024c5-3p-conv0' % Raw
    autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true)
    opts.arch = '1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
    autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true)
end

%% CIFAR-100
opts.cifar100 = true;
opts.n_train = 50e3;
opts.n_folds = 1;
opts.PCA_dim = 1000;
% No augmentation with RA
opts.arch = '1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'})

% Augmentation
opts.PCA_dim = 1500;
opts.arch = '1024c5-3p-conv0__32g-32ch-256c5-3p-conv0__32g-256ch-1024c5-3p-conv0' % Raw
autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true)
opts.arch = '1024c5-3p-conv0_3__32g-32ch-256c5-3p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
autocnn_cifar(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true)
