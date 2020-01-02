%% k-nearest neighbors
%% Finding best k for by instance k-nn
This code has two parts. Part 1, calculates the mean accuracy in estimating
wavelengths of test set using k in the range k=[0:20] in order to find 
the best k that yileds the highest accuracy. This part of the code outputs 
the excel file 'kMeans.xlsx'. The largest value in this file should belong 
to the optimal k. This part only needs to be execued onece to find optimal k. 
For our data it turns out to be k=7. 

The second part is the main part that for each set of training and test
set, estimates the wavelengths of test set and writes them  in the
'knn_byInstance_k7.xlsx' along with the truth-value wavelengths. Upon
finishing, it also outputs the elapsed time for testing the entire test
set.

Here is the details of the input data to this code. The training data, test
data and labels were initially extracted in MATLAB,so we kept them the way 
they were. Trainng set and test set are long matrices of 11 columns of 
transmittances values, one column per filter. The label files are single 
column vectors containing the labels corresponding to each row of the \
training/tests sets. The samples should be randomly shuffled before inputing 
to the code. The suffled data as .mat files are provided in the Data folder.