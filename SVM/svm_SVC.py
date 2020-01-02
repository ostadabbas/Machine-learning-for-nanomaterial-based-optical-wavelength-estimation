## Support vector machines, using Sklearn
# The user may change the file name ending: _linear if different kernel is employed.

from sklearn import svm
import joblib
import numpy as np
import xlsxwriter
import scipy.io as sio
import time
import os

# You can chose the kernel type, degree and tolerance parameters here
# kernel='rbf', tol=0.000001, degree=1
svm_lin_clf = svm.SVC(gamma='auto', kernel='linear', tol=1e-6)

if __name__ == '__main__':
    
    Restore = True

    #Path parameters
    save_PATH = './ckpt_svmT_files'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
        
    train_set = sio.loadmat('trainT.mat')['trainT']
    test_set = sio.loadmat('testT.mat')['testT']
    train_labels = sio.loadmat('trainT_labels.mat')['trainT_labels'].reshape(-1)
    test_labels = sio.loadmat('testT_labels.mat')['testT_labels'].reshape(-1)

    min_val = train_labels.min()
    train_labels = train_labels - min_val
    test_labels = test_labels - min_val

    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)
        
    PATH_Net = save_PATH + '/svm_lin_clf_linear_tole6'
        
    if Restore == False:
        print("Training...")
        
        #Load from previously trained file
        # Comment the 2 lines below if loading from a trained model
        print ("using trained model")
        svm_lin_clf = joblib.load(PATH_Net)
    
        #Train new model
        # Comment the 3 lines below if loading from a trained model
        print ("building new model")
        svm_lin_clf.fit(train_set, train_labels)
        joblib.dump(svm_lin_clf, PATH_Net)
        
        time_start = time.time()    
        pred_test = svm_lin_clf.predict(test_set)
        time_end = time.time() 
        pred_train = svm_lin_clf.predict(train_set)
        acc_test = np.sum(pred_test == test_labels) / test_set.shape[0] * 100
        acc_train = np.sum(pred_train == train_labels) / train_set.shape[0] * 100
        print ('Test_set accuracy : %d' % (acc_test))
        print ('Training_set accuracy : %d' % (acc_train))
        print ('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        
    if Restore:
        print("Testing...")
        
        svm_lin_clf = joblib.load(PATH_Net)
        print ("using trained model")
        
        time_start = time.time()
        pred_test = svm_lin_clf.predict(test_set)
        time_end = time.time()
        
        acc_test = np.sum(pred_test == test_labels) / test_set.shape[0] * 100
        print ('Test_set accuracy : %d' % (acc_test))
    
        
    Trials_estimation = pred_test + min_val
    Trials_labels = test_labels + min_val
    vals = [(Trials_labels[i], Trials_estimation[i]) for i in range (len(Trials_labels))]
    
    workbook = xlsxwriter.Workbook('Estimation_by_SVM_T_linear.xlsx')
    # This file contains two columns: real wavelengths and the estimated wavelegnths.
    worksheet = workbook.add_worksheet()
    col = 0

    for row, data in enumerate(vals):
        worksheet.write_row(row, col, data)
    workbook.close()
    
    print ('elapsed time (sec) : %0.2f' % ((time_end-time_start)))

    
    
    