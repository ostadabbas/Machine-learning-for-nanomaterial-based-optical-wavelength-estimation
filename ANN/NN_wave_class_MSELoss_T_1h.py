# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:11:14 2019

@author: Amir
"""

import torch, torch.nn as nn
import torch.utils.data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import xlsxwriter
import os
import time
import scipy.io as sio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))



class NN_classifier(nn.Module):
    def __init__(self, inp, h1, class_num):
        super().__init__()
        self.fc1 = nn.Linear(inp, h1, bias=True)
        self.fc2 = nn.Linear(h1, class_num, bias=True)
#        self.fc3 = nn.Linear(h2, class_num, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        fc1_out = self.tanh(self.fc1(x))
#        fc1_out = self.fc1(x)
        class_output = self.sigmoid(self.fc2(fc1_out))
#        fc2_out = self.tanh(self.fc2(fc1_out))
#        class_output = self.sigmoid(self.fc3(fc2_out)) # Why sigmoid?
        output = self.softmax(self.fc2(fc1_out))
    
        return class_output, output
 
#CELoss = torch.nn.CrossEntropyLoss(size_average = False, reduce = True)
mseLoss = torch.nn.MSELoss(size_average = False, reduce = True)

if __name__ == '__main__':
    
    Restore = True

    #Path parameters
    save_PATH = './ckpt_nnT_1h_files'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
    
    # load  Dataset
    train_set = sio.loadmat('./Data/trainT.mat')['trainT']
    test_set = sio.loadmat('./Data/testT_190812.mat')['testT_190812']
    train_labels = sio.loadmat('./Data/trainT_labels.mat')['trainT_labels'].reshape(-1)
    test_labels = sio.loadmat('./Data/testT_labels_190812.mat')['testT_labels_190812'].reshape(-1)
    
    min_val = train_labels.min()
#    mat = np.zeros(shape = (len(train_labels), 770)) # 770 is Class_num
#    for ii in range(len(train_labels)):  
#        mat[ii, train_labels[ii]-min_val] = 1            # 331 is 
#    train_labels = mat
    
#    mat = np.zeros(shape = (len(test_labels), 770))
#    for ii in range(len(test_labels)):
#        mat[ii, test_labels[ii]-min_val] = 1
#    test_labels = mat
        

    batch_size = 5000
    nData, inp = train_set.shape
    
#    max_value = train_set.max()
#    train_set = train_set / max_value
    train_set = torch.FloatTensor(train_set)
    train_labels = torch.LongTensor(train_labels.astype('double')- min_val)   ## .astype('double')
    
    training_set = [[train_set[i], train_labels[i]] for i in range(len(train_set))]
    
#    test_set = test_set / max_value
    test_set = torch.FloatTensor(test_set)
    test_set = test_set.to(device)
    test_labels = torch.LongTensor(test_labels.astype('double')- min_val) ## .astype('double')
    test_labels = test_labels.to(device)
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    lr = 1e-5
    h1 = 100
    class_num = 750
    epoch_num = 100000
    PATH_Net = save_PATH + '/Net_epoch%d' % (epoch_num)
    PATH_Opt = save_PATH + '/Opt_epoch%d' % (epoch_num)
    
    nn_classifier = NN_classifier(inp, h1, class_num)
    optimizer = optim.Adam(nn_classifier.parameters(), lr=lr)


    if Restore == False:
        print("Training...")
        
        # Comment out the line below if starting from scratch
        nn_classifier.load_state_dict(torch.load(PATH_Net)) 
        optimizer.load_state_dict(torch.load(PATH_Opt)) 
        
        for i in range(epoch_num):
            time_start = time.time()
            loss_value = 0.0
            acc_train = 0
            
            for batch_indx, data in enumerate(train_loader):
            
                # update AutoEncoder
                train_data, labels_data = data
                train_data = Variable(train_data).to(device)
                
#                see = torch.FloatTensor(labels_data)
                mat = np.zeros(shape = (batch_size, class_num))
                labels_data = np.array(labels_data)
                for j in range(batch_size):
                    mat[j, labels_data[j].astype('int16')] = 1
                labels_data = mat
                labels_data = torch.FloatTensor(labels_data)
                
                labels_data = Variable(labels_data).to(device)
                
                # zero the gradients
                optimizer.zero_grad()
            
                # get output from both modules  
                class_output, output = nn_classifier.forward(train_data)
                _, pred_idx = torch.max(output, dim = 1)
                _, labels_idx = torch.max(labels_data, dim = 1)
                acc_train += torch.sum(pred_idx == labels_idx).type('torch.FloatTensor')
#                acc_train += torch.sum(torch.abs(pred_idx - labels_data)/(labels_data + min_val))
                
                
#                back propagation
#                loss = CELoss(class_output, labels_data)
                loss = mseLoss(output, labels_data)
                loss.backward()
                optimizer.step()
    
                loss_value += loss.item()
                
            time_end = time.time()
            
#            _, output_test = nn_classifier.forward(test_set)
#            _, pred_idx = torch.max(output_test, dim = 1)
            _, output_test = nn_classifier.forward(test_set)
#            output_test = np.array(output_test[0])
            _, pred_idx = torch.max(output_test, dim = 1)
#            _, testlabel_idx = torch.max(test_labels, dim = 1)
            

            acc_test = torch.sum(pred_idx == test_labels).type('torch.FloatTensor') / test_set.size()[0] * 100
            acc_train_value = acc_train / train_set.size()[0] * 100
            
            print('elapsed time (min) : %0.2f' % ((time_end-time_start)/60))
            print('====> Epoch: %d Train_Loss : %0.8f | Train_Acc : %0.2f | Test_Acc : %0.2f'\
                  % ((i + 1),\
                     loss_value / len(train_loader.dataset),\
                     acc_train_value.item(),\
                     acc_test.item()))
            
            torch.save(nn_classifier.state_dict(), PATH_Net)
            torch.save(optimizer.state_dict(), PATH_Opt)

    if Restore:
        print("Testing...")
        
        nn_classifier.load_state_dict(torch.load(PATH_Net)) #, map_location=lambda storage, loc: storage))
        optimizer.load_state_dict(torch.load(PATH_Opt)) 
        time_start = time.time()
        _, output_test = nn_classifier.forward(test_set)
    
    _, pred_idx = torch.max(output_test, dim = 1)
    time_end = time.time()
#    _, testlabel_idx = torch.max(test_labels, dim = 1)
#    Trials_estimation = test_labels[pred_idx]  ## check this
    Trials_estimation = pred_idx.detach().numpy() + min_val
    Trials_labels = test_labels.detach().numpy() + min_val
    vals = [(Trials_labels[i], Trials_estimation[i]) for i in range(len(Trials_labels))]
        
    workbook = xlsxwriter.Workbook('Estimation_by_MSELoss_T_1h_190812.xlsx')
    worksheet = workbook.add_worksheet()
    col = 0
    
    for row, data in enumerate(vals):
        worksheet.write_row(row, col, data)
    workbook.close()
        
    
    print ('elapsed time (sec) : %0.3f' % ((time_end-time_start)))
    
#    rel_loss = 0
#    _, output = nn_classifier.forward(train_set)
#    patch_power = torch.sqrt(torch.mean(torch.pow(patches[i],2))).item()
#    error_power = torch.sqrt(torch.mean(torch.pow(patches[i]-reconstructed_output,2))).item()
#    #rel_loss += L2Loss.item() / patch_power
#    rel_loss += error_power / patch_power
#       
#    print(rel_loss/nPatches)
#    print('success!')