## Support vector machines, using Sklearn
This code trains a support vector machine model by desired parameters and kernel type, 
and finally outputs a file containing the estimated wavelengths as well as truth-value 
wavelengths of the test samples. The training data, test data and labels were initially 
extracted in MATLAB, so we kept them the way they were. Trainng set and test set are 
long matrices of 11 columns of transmittances values, one column per filter. The label 
files are single column vectors containing the labels corresponding to each row of the 
training/tests sets. The samples should be randomly shuffled before inputing to the code. 

This code can either start training the model from scracth or use already 
trained model, depending on whether the "Restore" value is set to False or True (shown below). 
If Restore == True, the code will load from a saved model in './ckpt_svmT_files', else, it will 
create the './ckpt_svmT_files' to save the parameter files. When the "fit" function has finished 
its job the "predict" function evaluates the estimated wavelengths of test samples and writes 
them in the file named 'Estimation_by_SVM_T_linear.xlsx'; it also outputs the elapsed time for 
testing the entire test set.
The user may change the file name ending: _linear if different kernel is employed.