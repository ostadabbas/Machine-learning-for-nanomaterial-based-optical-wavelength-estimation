## Bayesian inference, part 2:

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlsxwriter
import time

def load(fname):
    model = None
    with open(fname,'r') as f:
        model = json.load(f)
    model['waveD'] = {w:idx for idx,w in enumerate(model['wavelengths'])}
    model['means'] = np.array(model['means'])
    model['stds'] = np.array(model['stds'])
    return model

def get_row(model,lam):
    idx = model['waveD'][lam]
    mns = model['means'][idx,:]
    stds = model['stds'][idx,:]
    return mns,stds   

def gen_test(model, lam):
# generate a test value for a given frequency using recorded stats
    mns,std = get_row(model,lam)
    val = [np.random.normal(m,s) for m,s in zip(mns,std)]
    return val

def prob_obs_given_lambda(obs, model, lam):
    mns,std = get_row(model,lam)
    probs = [stats.norm.pdf(x,m,s) for x,m,s in zip(obs,mns,std)]
    prob = np.prod(probs)
    return prob

def full_prob(obs,model):
    waves = model['wavelengths']

    probs = [prob_obs_given_lambda(obs,model,w) for w in waves]
    probs = np.array(probs)
    probs = probs/np.sum(probs)
    return probs, waves

def plot_prob(obs, model):
    probs, waves = full_prob(obs,model)
    plt.figure()
    plt.plot(model['wavelengths'], probs, '.')
    plt.ylabel('Posterior Probability',fontname="Times New Roman",fontweight="bold")
    plt.xlabel('Wavelength (nm)',fontname="Times New Roman",fontweight="bold")
    plt.title('Posterior probability for real wavelength $\lambda$=700nm using 2 filters',fontname="Times New Roman",fontweight="bold")
    plt.grid(True)
    plt.show()
    plt.savefig("MAP analysis.pdf")
    
def test_trials():
    model = load("trans.json")
    waves = model['wavelengths']
    trials = np.genfromtxt('trialsT.csv', delimiter=',')
    trials = trials[:2,:]
    print(trials.shape)
    freqs = trials[:,0].astype(int)
    data = trials[:,1:] 
    
    def find_map(obs):
        prob, w = full_prob(obs,model) # w is not used
        lam = waves[np.argmax(prob)]
        plt.plot(prob)
        return lam
    def find_all():
        for idx,f,x in zip(range(len(freqs)),freqs,data):
            yield f,find_map(x)
            if (idx % 10) == 0:
                print("{} of {}".format(idx,len(freqs)))
    vals = list(find_all())
    return vals


def demo():
# "Call this function for:"
# By calling as demo() in console or command line, this function synthesises a test 
# sample of transmittances for the desired wavelength shown by waves=[some wavelength] 
# below; then performs Bayesian algorithm and plots the probability distributiob vs. 
# wavelength. The maximum value of this plot is the result of MAP estimation for the 
# input desired wavelength. It also saves the plot as "MAP analysis.pdf".
    model = load("trans.json")
    waves=[360]
    for w in waves:
        x = gen_test(model, w)
        plot_prob(x,model)

def return_vals():
    model = load("trans.json")
    waves=[1020]
    for w in waves:
        x = gen_test(model,w)
        probs, waves = full_prob(x,model)
    return probs, waves

def arg_max(input_waves):
# "Call this function for:"
# This function takes the list of user-defined walenghts (e.g. [400, 500, 600]), 
# synthesizes test samples for them, finds their estimated wavelengths, outputs
# the results in the screen as well as writes them in file "Argmax_train.xlsx".
    model = load("trans.json")
    ws = len(input_waves) # number of desired wavelengths to try
    num = 10              # How many times to try each wavelength
    max_probs = np.zeros(shape=(ws,num))
    wavelengths = np.array(model['wavelengths'])
    for i in range (ws):
        for j in range (num):
            x = gen_test(model, input_waves[i])
            probabilities, wave = full_prob(x,model) # wave is not being used
            max_probs[i,j] = wavelengths[np.argmax(probabilities)]
    workbook = xlsxwriter.Workbook("Argmax_train.xlsx")
    worksheet = workbook.add_worksheet()
    row = 1
    for col, data in enumerate(max_probs):
        worksheet.write_column(row, col, data)
    workbook.close()  
    return max_probs


input_waves = np.arange(1100,349,-1)  # Example of input waves that user can provide. 
# Here we have provided an array of all walengths in the training set, so calling 
# test samples for alll of these wavelengths, esrtimate their wavelength, and output 
# arg_max(input_waves) will synthesie on the screen.


def check_trials(): 
# "Call this function for:"
# This function estimates the wavelength of entire or a portion of the test set.
# While doing so, it shows the resuts at every 100 saples, and finally writes the 
# entire results in "Argmax_test.xlsx" file. This function also outputs the 
# elapsed time for testing the entire test set.
    model = load("trans.json")
    waves = model['wavelengths']
    trials = np.genfromtxt('trialsT.csv', delimiter=',')
    trials = trials[:,:] # Here we can set which portion of test set to be tested. The default is all.
    freqs = trials[:,0].astype(int)
    data = trials[:,1:]
    mns = model['means']
    stds = model['stds']
    stds_log = np.sum(np.log(stds), axis = 1)

    def find_map(obs):
        #pdb.set_trace()
        prob_li = -stds_log-1.0/2*np.sum(((mns-obs)/stds)**2, axis = 1)

        lam = waves[np.argmax(prob_li)]
        return lam
        
    def find_all():
        for idx,f,x in zip(range(len(freqs)),freqs,data):
            yield f,find_map(x)
            if (idx % 100) == 0:
                print("{} of {}".format(idx,len(freqs)))
                
    time_start = time.time()            
    vals = list(find_all())
    time_end = time.time()
    
    workbook = xlsxwriter.Workbook("Argmax_test.xlsx")
    # This file contains two columns: real wavelengths and the estimated wavelegnths.
    worksheet = workbook.add_worksheet()
    row = 1

    for col, data in enumerate(vals):
        worksheet.write_column(row, col, data)
    workbook.close()  
    
    print('elapsed time (sec) : %0.2f' % ((time_end-time_start)))
    return vals
    
    
    
    