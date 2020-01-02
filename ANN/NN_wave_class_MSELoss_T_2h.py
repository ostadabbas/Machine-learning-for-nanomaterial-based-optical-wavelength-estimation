
import torch, torch.nn as nn
import torch.utils.data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import xlsxwriter
import os
import time
import scipy.io as sio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

class NN_classifier(nn.Module):
    def __init__(self, inp, h1, h2, class_num):
        super().__init__()
        self.fc1 = nn.Linear(inp, h1, bias=True)
        self.fc2 = nn.Linear(h1, h2, bias=True)
        self.fc3 = nn.Linear(h2, class_num, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        fc1_out = self.tanh(self.fc1(x))
        fc2_out = self.tanh(self.fc2(fc1_out))
        class_output = self.sigmoid(self.fc3(fc2_out)) 
        output = self.softmax(self.fc3(fc2_out))
    
        return class_output, output

mseLoss = torch.nn.MSELoss(size_average = False, reduce = True)

if __name__ == '__main__':
    
    Restore = True

    #Path parameters
    save_PATH = './ckpt_nnT_2h_files'
    if not os.path.exists(save_PATH):
        os.makedirs(save_PATH)
    
    # load  Dataset
    train_set = sio.loadmat('trainT.mat')['trainT']
    test_set = sio.loadmat('testT.mat')['testT']
    train_labels = sio.loadmat('trainT_labels.mat')['trainT_labels'].reshape(-1)
    test_labels = sio.loadmat('testT_labels.mat')['testT_labels'].reshape(-1)
    
    min_val = train_labels.min()
        

    batch_size = 5000
    nData, inp = train_set.shape

    train_set = torch.FloatTensor(train_set)
    train_labels = torch.LongTensor(train_labels.astype('double')- min_val)
    training_set = [[train_set[i], train_labels[i]] for i in range(len(train_set))]
    
    test_set = torch.FloatTensor(test_set)
    test_set = test_set.to(device)
    test_labels = torch.LongTensor(test_labels.astype('double')- min_val)
    test_labels = test_labels.to(device)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    lr = 1e-4  # Learning rate. Should be picked careffully for code to diverge
    h1 = 100   # First hidden layer size
    h2 = 400   # Second hidden layer size
    class_num = 750 # Output layer size equal to number of distinct wavelengths
    epoch_num = 100000 # Number of training iterations
    PATH_Net = save_PATH + '/Net_epoch%d' % (epoch_num) # Paths to save the state of the system in case
    PATH_Opt = save_PATH + '/Opt_epoch%d' % (epoch_num) # the trainig stops unexoectedly or by force
    
    nn_classifier = NN_classifier(inp, h1, h2, class_num)
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
                
                optimizer.zero_grad()
            
                class_output, output = nn_classifier.forward(train_data)
                _, pred_idx = torch.max(output, dim = 1)
                _, labels_idx = torch.max(labels_data, dim = 1)
                acc_train += torch.sum(pred_idx == labels_idx).type('torch.FloatTensor')
                
                loss = mseLoss(output, labels_data)
                loss.backward()
                optimizer.step()
                loss_value += loss.item()
                
            time_end = time.time()
            _, output_test = nn_classifier.forward(test_set)
            _, pred_idx = torch.max(output_test, dim = 1)            

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
                
        nn_classifier.load_state_dict(torch.load(PATH_Net)) 
        optimizer.load_state_dict(torch.load(PATH_Opt)) 
        time_start = time.time()
        _, output_test = nn_classifier.forward(test_set)
    
    _, pred_idx = torch.max(output_test, dim = 1)
    time_end = time.time()
    Trials_estimation = pred_idx.detach().numpy() + min_val
    Trials_labels = test_labels.detach().numpy() + min_val
    vals = [(Trials_labels[i], Trials_estimation[i]) for i in range(len(Trials_labels))]
    
    workbook = xlsxwriter.Workbook('Estimation_by_MSELoss_T_2h.xlsx')
    # This file contains two columns: real wavelengths and the estimated wavelegnths.
    worksheet = workbook.add_worksheet()
    col = 0
    
    for row, data in enumerate(vals):
        worksheet.write_row(row, col, data)
    workbook.close()
    print ('elapsed time (sec) : %0.3f' % ((time_end-time_start)))
    