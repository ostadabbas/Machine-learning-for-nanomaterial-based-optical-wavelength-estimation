import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze light sensor data")

    parser.add_argument('-l', '--list-wavelengths', action='store_true', default=False, help="list all wavelengths and exit")
    parser.add_argument('-t', '--tests-per-wave', metavar = 'N', type=int, help="Generated values per wavelength value", default=10)
    parser.add_argument('-m', '--min', type=float, default=None, help="Minimum wavelength")
    parser.add_argument('-x', '--max', type=float, default=None, help="Maximum wavelength")
    parser.add_argument('wave', nargs="*", type=int, default=None, help="Individual additional wavelengths")

    return parser.parse_args()


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
def log_prob_obs_given_lambda(obs, model, lam):
    mns,std = get_row(model,lam)
    probs = [stats.norm.logpdf(x,m,s) for x,m,s in zip(obs,mns,std)]
    prob = np.sum(probs)
    return prob

def full_prob(obs,model):
    waves = model['wavelengths']
    probs = [prob_obs_given_lambda(obs,model,w) for w in waves]
    probs = np.array(probs)
    probs = probs/np.sum(probs)
    return probs
def full_prob_log(obs,model):
    waves = model['wavelengths']
    probs = [log_prob_obs_given_lambda(obs,model,w) for w in waves]
    probs = np.array(probs)
    #probs = probs/np.sum(probs)
    return probs
def plot_prob(obs, model):
    probs = full_prob(obs,model)
    plt.figure()
    plt.plot(model['wavelengths'], probs, '.')
    plt.show()

def errNorm(mapF, goldF, ord):
    l = len(mapF)
    err = np.power(np.sum(np.abs(np.power(mapF - goldF, ord))), 1/ord)
    return err/l

class ProcTests:
    @staticmethod
    def loadModel():
        model = load('energy.json')
        waves = np.array(model['wavelengths'])
        sensors = model['sensors']
        return model, waves, sensors

    def __init__(self,errOrd=1):
        self.model, self.waves, self.sensors = self.loadModel()

        self.errOrd = errOrd
    def genData(self, nTests):
        tfreqs,tests = [],[]
        for lam in self.waves:
            for idx in range(nTests):
                tfreqs.append(lam)
                tests.append( gen_test(self.model, lam) )
        self.setData(tfreqs,tests)
    def loadTrials(self):
        trials = np.genfromtxt('Trials_190224.csv', delimiter=',')
        self.setData(trials[:,0], trials[:,1:])
    def setData(self, freqs, data):
        self.tfreqs = np.array(freqs).astype(int)
        self.tdata = np.array(data)

        #--- for each test val/sensor generate log prob ---
        tprobs = np.zeros( (len(self.tdata), len(self.waves), len(self.sensors)) )
        for widx,lam in enumerate(self.waves):
            mns, std = get_row(self.model, lam)
            for sidx in range(len(self.sensors)):
                tprobs[:, widx, sidx] = stats.norm.logpdf(self.tdata[:,sidx], mns[sidx], std[sidx])
        self.tprobs = tprobs

    def getAllSensors(self):
        return np.arange(len(self.sensors)).astype(int)
    def getAllWaves(self):
        waves = list(set(self.tfreqs))
        return np.copy(np.array(waves))
    def predFreq(self, sensors=None, allowed_waves=None):
        if allowed_waves is None:
            allowed_waves = self.waves

        def getWMap(all_waves):
            awset = set(allowed_waves)
            sl = np.array([w in awset for w in all_waves]).astype(bool)
            return sl
        tfM = getWMap(self.tfreqs)
        wM = getWMap(self.waves)
        tprobs = self.tprobs[tfM, :,:]
        tprobs = tprobs[:,wM,:]

        tprobs2 = tprobs[:,:,sensors] if (sensors is not None) else tprobs
        ttprobs = np.sum(tprobs2, axis=2)
        mapi = np.argmax(ttprobs, axis=1)
        waves = self.waves[wM]
        return waves[mapi], self.tfreqs[tfM]
    def calcErr(self,sensors=None, allowed_waves=None):
        pfreqs, tfreqs = self.predFreq(sensors, allowed_waves)
        err = errNorm(pfreqs, tfreqs, self.errOrd)
        return err
    def plotErr(self, sensors=None, allowed_waves=None, title=None):
        pfreqs, tfreqs = self.predFreq(sensors, allowed_waves)

        vals = {}
        for tf,pf in zip(tfreqs, pfreqs):
            if tf not in vals:
                vals[tf] = []
            vals[tf].append(tf-pf)
        vals = {key: np.mean(np.abs(v)) for key, v in vals.items()}
        plt.figure()
        kys = np.array(list(vals.keys()))
        plt.plot(kys, vals.values(), '.')
        if title is not None:
            t2 = '{}: {}'.format(title, getWaveStr(allowed_waves))
            plt.title(t2)
        plt.xlabel('wavelengths')
        plt.ylabel('average absolute error')



def test_trials():
    model = load('energy.json')
    waves = model['wavelengths']

    trials = np.genfromtxt('Trials_190224.csv', delimiter=',')
#    trials = trials[:102,:]
    freqs = trials[:,0].astype(int)
    data = trials[:,1:]

    print(len(waves))

    def find_map(obs):
        prob = full_prob_log(obs,model)
        lam = waves[np.argmax(prob)]
        return lam
    def find_all():
        for f,x in zip(freqs,data):
            fm = find_map(x)
            yield f,fm
    total=0
    for idx, dt in enumerate(find_all()):
        f,fm = dt
        total += abs(f-fm)
        if (idx % 10) == 0:
            print("{} of {}, {}".format(idx,len(freqs), total/(idx+1)))
    return vals
def demo():
#    model = load('trans.json')
    model = load('energy.json')
    waves=[400,500,600,700,800,900,1000,1100]
    for w in waves:
        x = gen_test(model, w)
        plot_prob(x,model)

def find_sens(waves, sensors, pt, pt2):
    print(len(waves))
    sensors = np.array(sensors)

    all_sensors = pt.getAllSensors()
    def search(sensors, applyF):
        errs = []
        for idx,s in enumerate(sensors):
            errs.append(pt.calcErr(applyF(sensors, idx), waves))
        idx = np.argmin(errs)
        return applyF(sensors,idx)
    def sub1(sensors):
        return search(sensors, lambda s,idx: np.delete(s,idx))
    def add1(sensors):
        def invert(sens):
            return list(set(all_sensors) - set(sens))
        return search(invert(sensors), lambda s,idx: invert(np.delete(s,idx)))


    chosen = all_sensors
    pt2.plotErr(chosen, waves, title='Test data')
    pt.plotErr(chosen, waves, title='Generated data')
    while len(chosen) > 0:
        print(sensors[chosen], pt.calcErr(chosen, waves), pt2.calcErr(chosen, waves))
        # pt.plotErr(chosen)
        chosen = sub1(chosen)
        # chosen = sub1(chosen)
        # chosen = add1(chosen)
def getWaveStr(waves):
    waves = sorted(waves)
    out = []
    for k, g in itertools.groupby(enumerate(waves), lambda iv:iv[0]-iv[1]):
        gg = [v[1] for v in g]
        if len(gg) > 1:
            out.append( '{}-{}'.format(min(gg), max(gg)) )
        elif len(gg) == 1:
            out.append( '{}'.format(gg[0]) )
    return ', '.join(out)


if __name__ == "__main__":
    args = parse_args()

    md, waves, sensors = ProcTests.loadModel()
    if len(args.wave) > 0:
        sw1 = set(waves)
        sw2 = set(args.wave)
        waves = sw1.intersection(sw2)
        waves = np.array(list(waves))

    if args.min is not None:
        waves = waves[waves >= args.min]
    if args.max is not None:
        waves = waves[waves <= args.max]

    #--- list wavelengths and exit if requested ---
    if args.list_wavelengths:
        print(getWaveStr(waves))
        exit()

    nvals = args.tests_per_wave
    genData = ProcTests()
    genData.genData(nvals)
    testData = ProcTests()
    testData.loadTrials()

    print(getWaveStr(waves))
    find_sens(waves, sensors, genData, testData)

    plt.show()
