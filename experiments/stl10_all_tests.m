% Experiments on the STL-10 test set

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

%% Comparison to Exemplar-CNN
archs = {[64,128,256],[92,256,512],[256,1024,2048]};
RAs = {{'0_3','0_3','0_2'},{'0','0','0'}}; % with RA and without (Raw)
for arch = archs
    for RA = RAs
        if (arch{1}(1) == 64)
            opts.arch = sprintf('%dc7-5p-conv%s__1g-%dch-%dc5-4p-conv%s__1g-%dch-%dc5-3p-conv%s', arch{1}(1),RA{1}{1},arch{1}(1),arch{1}(2),RA{1}{2},arch{1}(2),arch{1}(3),RA{1}{3})
            autocnn_stl10(opts) % no augmentation
            autocnn_stl10(opts, 'conv_norm', '') % also compute without Rootsift
        end
        opts.arch = sprintf('%dc7-4p-conv%s__1g-%dch-%dc5-4p-conv%s__1g-%dch-%dc5-3p-conv%s', arch{1}(1),RA{1}{1},arch{1}(1),arch{1}(2),RA{1}{2},arch{1}(2),arch{1}(3),RA{1}{3})
        opts.crop_repeat = 10; % for data augmentation with crops
        autocnn_stl10(opts, 'augment', true, 'crop', {72,0,0})
    end
end

%% Final tests with AutoCNN-L32
opts.gtsvm = gtsvm_path;
opts.liblinear = '';
opts.n_train = 1e3;
opts.n_folds = 10;
opts.PCA_dim = 1000;
% no augmentation with RA
opts.arch = '1024c7-5p-conv0_3__32g-32ch-256c5-4p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
autocnn_stl10(opts, 'learning_method', {'kmeans','pca','pca'})

% Augmentation
opts.PCA_dim = 1500;
opts.crop_repeat = 20;
opts.arch = '1024c7-4p-conv0__32g-32ch-256c5-4p-conv0__32g-256ch-1024c5-3p-conv0' % Raw
autocnn_stl10(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true, 'crop', {72,0,0})
opts.arch = '1024c7-4p-conv0_3__32g-32ch-256c5-4p-conv0_2__32g-256ch-1024c5-3p-conv0_2' % RA
autocnn_stl10(opts, 'learning_method', {'kmeans','pca','pca'}, 'augment', true, 'crop', {72,0,0})