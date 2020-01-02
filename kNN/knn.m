%% k-nearest neighbors

%% Finding best k for by instance k-nn

clear; close all; clc
fprintf('Starting to do prediction by knn method\n');

load('trainT.mat');
load('TrainT_labels.mat');
load('testT.mat');
load('testT_labels.mat');

[num_test, ~] = size(testT_labels);
Predictions = zeros(size(testT_labels));
[m, ~] = size(trainT);

kMeans = zeros(20,1);
for k = 1:20
    fprintf('Calculating for k = %d\n', k);
    Predictions = zeros(size(testT_labels));
    for ii = 1:num_test
        trial = testT(ii, :);
        pred = zeros(m,2);
        pred(:,1) = trainT_labels(:);
        for jj = 1:m
            t = trainT(jj, :);
            pred(jj, 2) = sum((t - trial).^2, 2);
        end
        P = sortrows(pred, 2);
        results = P(1:k, 1);
        Predictions(ii) = mode(results);
    end
    kMeans(k) = mean(Predictions == testT_labels);
end
xlswrite('kMeans.xlsx', [[1:20]', kMeans]); % The optimal value of k will 
% show the maximum kMeans value.

%% Doing knn by k = 7 for test samples

clear; close all; clc

fprintf('Starting to do prediction by knn method\n');

load('trainT.mat');
load('TrainT_labels.mat');
load('testT.mat');
load('testT_labels.mat');

k = 7;
[num_test, ~] = size(testT_labels);
Predictions = zeros(size(testT_labels));

[m, ~] = size(trainT);

tic;
for ii = 1:num_test
    trial = testT(ii, :);
    pred = zeros(m,2);
    pred(:,1) = trainT_labels(:);
    for jj = 1:m
        t = trainT(jj, :);
        pred(jj, 2) = sum((t - trial).^2, 2);
    end
    P = sortrows(pred, 2);
    results = P(1:k, 1);
    Predictions(ii) = mode(results);
end
toc;

xlswrite('knn_byInstance_k7.xlsx', [testT_labels, Predictions]);
% This file contains two columns: real wavelengths and the estimated
% wavelegnths.