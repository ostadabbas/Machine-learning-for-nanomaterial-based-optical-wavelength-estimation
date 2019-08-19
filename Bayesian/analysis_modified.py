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
    trials = np.genfromtxt('trialsT_New.csv', delimiter=',')
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
#    model = load('trans_prime.json')
#    model = load('trans.json')
#    waves=[400,500,600,700,800,900,1000,1100]
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

#input_waves = [350,375,400,425,450,475,500,525,550,575,600,625,637,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075]
input_waves = np.arange(1100,349,-1)
    
#for entire spectrum:
#    model = load("trans_prime.json")
#    input_waves = model['wavelengths']
    
def check_trials(): # 
    model = load("trans.json")
    waves = model['wavelengths']
    trials = np.genfromtxt('trialsT_New.csv', delimiter=',')
    trials = trials[:,:]
    freqs = trials[:,0].astype(int)
    data = trials[:,1:]
    mns = model['means']
    stds = model['stds']
#    wave_st = 350 
    
    def find_map(obs):
        # make a talbe  
        # from obs, 11 
        # 
        #  for wave in waves: for T in obs: get mn std, get  prob  add to list prob_li [N x 11]
        # make prob 
        prob_li = []
        i = 0 
        for mean_wv, std_wv in zip(mns,stds): 
#            wave = i+ wave_st
            prob_wv_li = []
#            chkTm_st = time.time()
            for T, mean, std in zip(obs, mean_wv, std_wv):
                probT = stats.norm.pdf(T,mean,std)  # time consued about 0.1 ms for each normal dist
                prob_wv_li.append(probT)    # 11 ft pro
            prob_li.append(np.prod(prob_wv_li))
#            if i == 300:
#                print('each wv quiry takes time', time.time() - chkTm_st)
#            i +=1 
        lam = waves[np.argmax(prob_li)]
        return lam
        
#        prob, w = full_prob(obs,model) # w is not used
#        lam = waves[np.argmax(prob)]
#        return lam
    def find_all():
        for idx,f,x in zip(range(len(freqs)),freqs,data):
            yield f,find_map(x)
            if (idx % 10) == 0:
                print("{} of {}".format(idx,len(freqs)))
                
    time_start = time.time()            
    vals = list(find_all())
    time_end = time.time()
    
    workbook = xlsxwriter.Workbook("Argmax_test_Liu.xlsx")
    worksheet = workbook.add_worksheet()
    row = 1

    for col, data in enumerate(vals):
        worksheet.write_column(row, col, data)
    workbook.close()  
    
    print('elapsed time (sec) : %0.2f' % ((time_end-time_start)))
    return vals
    
    
    
    