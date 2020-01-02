## Artificial neural networks, using Torch


NN_wave_class_MSELoss_T_1h.py:
------------------------------
This code creates a 3-layer fully connected neural network architecture, an input layer, 
a hidden layer and an output laye, to train the model, and finally outputs a file containing 
the estimated wavelengths as well as truth-value wavelengths of the test samples. 
The training data, test data and labels were initially extracted in MATLAB, so we kept them 
the way they were. Trainng set and test set are long matrices of 11 columns of transmittances 
values, one column per filter. The label files are single column vectors containing the labels 
corresponding to each row of the training/tests sets. The samples should be randomly shuffled 
before inputing to the code. The suffled data as ".mat" files are provided in the Data folder.

This code can either start training the model from scracth or use already trained model, 
depending on whether the "Restore" value is set to False or True (shown below). 
If Restore == True, the code will load from a saved model in './ckpt_nnT_1h_files', else, it 
will create the './ckpt_nnT_1h_files' to save the parameter files. The code also can run on 
GPU as well as CPU if GPU is available. While training, the code will output the training 
and testing accuracy. When the number of training loops shown by epoch_num is finished the code 
evaluates the estimated wavelengths of test samples and writes them in the file named
'Estimation_by_MSELoss_T_1h.xlsx'; it also outputs the elapsed time for testing the entire
test set.



NN_wave_class_MSELoss_T_2h.py:
------------------------------
This code creates a 4-layer fully connected neural network architecture, an input layer, 
two hidden layers and an output laye, to train the model, and finally outputs a file
containing the estimated wavelengths as well as truth-value wavelengths of the 
test samples. The training data, test data and labels were initially extracted in MATLAB,
so we kept them the way they were. Trainng set and test set are long matrices of 11 
columns of transmittances values, one column per filter. The label files are single column
vectors containing the labels corresponding to each row of the training/tests sets.
The samples should be randomly shuffled before inputing to the code. The suffled data as ".mat"
files are provided in the Data folder.

This code can either start training the model from scracth or use already 
trained model, depending on whether the "Restore" value is set to False or True (shown below). 
If Restore == True, the code will load from a saved model in './ckpt_nnT_2h_files', else, it 
will create the './ckpt_nnT_2h_files' to save the parameter files. The code also can run on 
GPU as well as CPU if GPU is available. While training, the code will output the training 
and testing accuracy. When the number of training loops shown by epoch_num is finished the code 
evaluates the estimated wavelengths of test samples and writes them in the file named
'Estimation_by_MSELoss_T_2h.xlsx'; it also outputs the elapsed time for testing the entire
test set.