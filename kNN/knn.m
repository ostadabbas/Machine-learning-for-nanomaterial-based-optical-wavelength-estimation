
%% Run from command line:
%% matlab -nodesktop -nosplash -r "foo"
%--------------------------------------

% %% Reading all E's into one file 
clear; close all; clc
% 
%% Parameters
filters = [1,2,3,4,5,6,7,8,9,10,11];
sofar = length(filters);
wave_range = 1100-350; 
wave = [1100:-1:351]';
len = wave_range;
%numspec = 500;   % Number of energy spectra
num_T = 100; % num_T is number of T spectra per filter
numtest = 10;   % Number of test spectra
tcount = 100; % Keep this number of spectra per filter



%% Doing knn by k = 7

clear; close all; clc
tic;
fprintf('Starting to do prediction by knn method\n');
cd 'Data';

% which_data = 'half';
% train_set = load(strcat('trainT_half', which_data, '.mat'), strcat('trainT_half', which_data));
% load(strcat('TrainT_labels_half', which_data, '.mat'));
% load(strcat('testT', '.mat'));
% load(strcat('testT_labels', '.mat'));

load('trainT_half.mat');
load('TrainT_labels_half.mat');
load('testT.mat');
load('testT_labels.mat');
cd ..

k = 7;
[num_test, ~] = size(testT_labels);
Predictions = zeros(size(testT_labels));

[m, ~] = size(trainT_half);

for ii = 1:num_test
    trial = testT(ii, :);
    pred = zeros(m,2);
    pred(:,1) = trainT_labels_half(:);
    for jj = 1:m
        t = trainT_half(jj, :);
        pred(jj, 2) = sum((t - trial).^2, 2);
    end
    P = sortrows(pred, 2);
    results = P(1:k, 1);
    Predictions(ii) = mode(results);
%     ii
end

xlswrite('knn_byInstance_k7_half.xlsx', [testT_labels, Predictions]);
toc;



%% knn Prediction by centroids
clear; close all; clc
tic;
numtest = 10;
fprintf('Starting to do prediction by knn method\n');
% cd 'C:\Users\Davoud\Dropbox\Reseach\Davoud-UV-VIS\Glass\'
T_avg = xlsread('T_avg_half.xlsx','T_avg_half');
waves = T_avg(:,1);
T_avg = T_avg(:,2:end);                          

[len, sofar] = size(T_avg);

testT = csvread('trialsT.csv');
test_labels = testT(:,1);
testT = testT(:,2:end);      
Predictions = zeros(size(test_labels)); 

for ii = 1:length(test_labels)
    trial = testT(ii, :);
    pred = zeros(len,1);
    for jj = 1:len
        t = T_avg(jj, :);
        pred(jj) = sum((t - trial).^2, 2);
    end
    [~, ind] = min(pred(:)); % prediction by min of squares
    Predictions(ii) = waves(ind);
%     ii
end

% fprintf('Writing predictions using original averaged data.xlsx\n');
xlswrite('knn_centroid_half.xlsx', [test_labels, Predictions]);
% xlswrite('knn_Predictions.xlsx', test_labels, 'knn', 'A2');
% fprintf('Prediction done; you can now open the mentioned excel files\n')
% fprintf('-----------------------------------------------------------\n');
% fprintf('-----------------------------------------------------------\n');
toc;


%% Finding best k for by instance k-nn

clear; close all; clc
tic;
fprintf('Starting to do prediction by knn method\n');
cd 'Data';

% which_data = 'half';
% train_set = load(strcat('trainT_', which_data, '.mat'), strcat('trainT_', which_data));
% load(strcat('TrainT_labels_', which_data, '.mat'));
% load(strcat('testT', '.mat'));
% load(strcat('testT_labels', '.mat'));

load('trainT_foneifth.mat');
load('TrainT_labels_foneifth.mat');
load('testT.mat');
load('testT_labels.mat');
cd ..

[num_test, ~] = size(testT_labels);
Predictions = zeros(size(testT_labels));
[m, ~] = size(trainT_foneifth);

kMeans = zeros(20,1);
for k = 1:20
    fprintf('Calculating for k = %d\n', k);
    Predictions = zeros(size(testT_labels));
    for ii = 1:num_test
        trial = testT(ii, :);
        pred = zeros(m,2);
        pred(:,1) = trainT_labels_foneifth(:);
        for jj = 1:m
            t = trainT_foneifth(jj, :);
            pred(jj, 2) = sum((t - trial).^2, 2);
        end
        P = sortrows(pred, 2);
        results = P(1:k, 1);
        Predictions(ii) = mode(results);
    end
    kMeans(k) = mean(Predictions == testT_labels);
end
xlswrite('kMeans.xlsx', [[1:20]', kMeans]);
toc;


%% Doing knn by k = 7 for NEW test samples

clear; close all; clc

fprintf('Starting to do prediction by knn method\n');
cd 'Data';

load('trainT.mat');
load('TrainT_labels.mat');
load('testT_New.mat');
load('testT_labels_New.mat');
cd ..

k = 7;
[num_test, ~] = size(testT_labels_New);
Predictions = zeros(size(testT_labels_New));

[m, ~] = size(trainT);

tic;
for ii = 1:num_test
    trial = testT_New(ii, :);
    pred = zeros(m,2);
    pred(:,1) = trainT_labels(:);
    for jj = 1:m
        t = trainT(jj, :);
%         tic
        pred(jj, 2) = sum((t - trial).^2, 2);
%         toc
    end
    P = sortrows(pred, 2);
    results = P(1:k, 1);
    Predictions(ii) = mode(results);
%     ii
end
toc;

xlswrite('knn_byInstance_k7_book.xlsx', [testT_labels_New, Predictions]);

%%
tic
normpdf(0.9,1.1,0.001)
toc
