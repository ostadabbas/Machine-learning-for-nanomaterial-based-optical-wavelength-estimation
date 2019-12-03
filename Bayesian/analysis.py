#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:08:44 2019

@author: davoud
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlsxwriter
import time
import pdb

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

# generate a test value for a given frequency using recorded stats
def gen_test(model, lam):
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
#    plt.figure()
#    plt.plot(waves,probs, '.')
#    plt.show()
    return probs, waves

def plot_prob(obs, model):
    probs, waves = full_prob(obs,model)
#    waves = model['wavelengths']
    plt.figure()
    plt.plot(model['wavelengths'], probs, '.')
    plt.ylabel('Posterior Probability',fontname="Times New Roman",fontweight="bold")
    plt.xlabel('Wavelength (nm)',fontname="Times New Roman",fontweight="bold")
    plt.title('Posterior probability for real wavelength $\lambda$=700nm using 2 filters',fontname="Times New Roman",fontweight="bold")
    plt.grid(True)
#    plt.show()
    plt.savefig("MAP analysis.pdf")
    
def test_trials():
    model = load("trans.json")
    waves = model['wavelengths']
    trials = np.genfromtxt('trialsT_191001.csv', delimiter=',')
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
    model = load("trans.json")
    waves=[360]
    for w in waves:
        x = gen_test(model, w)
#        x = test_trials()
        plot_prob(x,model)
        #plt.savefig('exported.pdf')
#    pr = full_prob(x,model)
        
    
        
def return_vals():
    model = load("trans.json")
    waves=[1020]
    for w in waves:
        x = gen_test(model,w)
        probs, waves = full_prob(x,model)
    return probs, waves

def arg_max(input_waves):
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

input_waves = np.arange(1100,349,-1)

    
def check_trials(): # 
    model = load("trans.json")
    waves = model['wavelengths']
    trials = np.genfromtxt('trialsT_191001.csv', delimiter=',')
#    howmany = 200
    trials = trials[:,:]
    freqs = trials[:,0].astype(int)
    data = trials[:,1:]
    mns = model['means']
    stds = model['stds']
    stds_log = np.sum(np.log(stds), axis = 1)
#    wave_st = 350 
    
#    pdb.set_trace()
    def find_map(obs):
        #pdb.set_trace()
        prob_li = -stds_log-1.0/2*np.sum(((mns-obs)/stds)**2, axis = 1)

        lam = waves[np.argmax(prob_li)]
        return lam
        
    def find_all():
        for idx,f,x in zip(range(len(freqs)),freqs,data):
            yield f,find_map(x)
#            if (idx % 100) == 0:
#                print("{} of {}".format(idx,len(freqs)))
                
    time_start = time.time()            
    vals = list(find_all())
    time_end = time.time()
    
#    workbook = xlsxwriter.Workbook("Argmax_191001_vectorized.xlsx")
#    worksheet = workbook.add_worksheet()
#    row = 1
#
#    for col, data in enumerate(vals):
#        worksheet.write_column(row, col, data)
#    workbook.close()  
    
    print('elapsed time (sec) : %0.2f' % ((time_end-time_start)))
#    return vals
    
    
    
    