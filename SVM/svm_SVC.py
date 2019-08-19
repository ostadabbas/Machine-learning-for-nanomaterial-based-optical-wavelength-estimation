# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:24:28 2019

@author: hejazi
"""

from sklearn import svm
from sklearn.externals import joblib
import numpy as np
# %matplotlib inline
import xlsxwriter
import scipy.io as sio
import time
import os

# kernel='linear', tol=0.000001, degree=1
svm_lin_clf = svm.SVC(gamma='auto', kernel='linear', tol=0.000001)

if __name__ == '__main__':
    
    Restore = True  

    #Path parameters
    save_PATH = './ckpt_svmT_files'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
        
    train_set = sio.loadmat('./Data/trainT.mat')['trainT']
    test_set = sio.loadmat('./Data/testT_New.mat')['testT_New']
    train_labels = sio.loadmat('./Data/trainT_labels.mat')['trainT_labels'].reshape(-1)
    test_labels = sio.loadmat('./Data/testT_labels_New.mat')['testT_labels_New'].reshape(-1)

    min_val = train_labels.min()
    train_labels = train_labels - min_val
    test_labels = test_labels - min_val

    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)
        
    PATH_Net = save_PATH + '/svm_lin_clf_linear_tole_6'
        
    if Restore == True:
        print("Training...")
    
        #Load from previous;y trained file
#        svm_lin_clf = joblib.load(PATH_Net)
#        print ("using trained model")
    
        print ("building new model")
        svm_lin_clf.fit(train_set, train_labels)
        joblib.dump(svm_lin_clf, PATH_Net)
        
            
        pred_test = svm_lin_clf.predict(test_set)
        pred_train = svm_lin_clf.predict(train_set)
#        print (pred == test_labels)
#        print (np.sum(pred == test_labels))
        acc_test = np.sum(pred_test == test_labels) / test_set.shape[0] * 100
        acc_train = np.sum(pred_train == train_labels) / train_set.shape[0] * 100
        print ('Test_set accuracy : %d' % (acc_test))
        print ('Training_set accuracy : %d' % (acc_train))
        time_end = time.time() 
        print ('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        
    if Restore:
        print("Testing...")
        
        svm_lin_clf = joblib.load(PATH_Net)
        print ("using trained model")
        
        time_start = time.time()
        pred_test = svm_lin_clf.predict(test_set)
        time_end = time.time()
        
#        pred_train = svm_lin_clf.predict(train_set)
        acc_test = np.sum(pred_test == test_labels) / test_set.shape[0] * 100
#        acc_train = np.sum(pred_train == train_labels) / train_set.shape[0] * 100
        print ('Test_set accuracy : %d' % (acc_test))
#        print ('Training_set accuracy : %d' % (acc_train))
    
        
        
    Trials_estimation = pred_test + min_val
    Trials_labels = test_labels + min_val
    vals = [(Trials_labels[i], Trials_estimation[i]) for i in range (len(Trials_labels))]
    
    workbook = xlsxwriter.Workbook('Estimation_by_SVM_T_New2.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0

    for row, data in enumerate(vals):
        worksheet.write_row(row, col, data)
    workbook.close()
    
    print ('elapsed time (sec) : %0.2f' % ((time_end-time_start)))

    
    
    