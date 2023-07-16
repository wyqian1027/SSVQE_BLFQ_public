import numpy as np
import os, re
from scipy.special import gamma, factorial, binom
from collections import defaultdict

# load spectroscopy/eigenvalue file
def load_Ham_data(f_path):
    output = []
    with open(f_path, "r") as f:
        output = f.readlines()
    return np.array([list(map(float, re.split('\s+', line.strip()))) for line in output])

def load_Ham_data_by_threshold(f_path, abs_threshold=1e-8):
    output = []
    with open(f_path, "r") as f:
        output = f.readlines()
    Ham = np.array([list(map(float, re.split('\s+', line.strip()))) for line in output])
    mask = np.abs(Ham) <= abs_threshold
    Ham[mask] = 0
    return Ham

def load_spectrum(f_path):
    data = []
    with open(f_path, "r") as f:
        data = f.readlines()
    output = []
    for line in data[1:]:
        output.append(list(map(float, (re.split('\s+', line.strip())))))
    return np.array(output)[:,5]

def load_evect_data(f_path):
    with open(f_path, "r") as f:
        output = f.readlines()
    return np.array([list(map(float, re.split('\s+', line.strip()))) for line in output])

# compute decay constants
def BigC(l, mu):
    ''' case of alpha = beta = mu '''
    l = int(l)
    front = np.sqrt((2*l+2*mu+1) / factorial(l) / gamma(l+2*mu+1)) * gamma((mu+3)/2)
    s = 0
    for k in range(0, l+1):
        s += binom(l, k)*(-1)**k*gamma(l+k+2*mu+1)/gamma(k+mu+1)*gamma(k+mu/2+3/2)/gamma(k+mu+3)
    return front*s
    
def get_decay_coeff(mf, kappa, Nc=3, n=0, l=0):
    mu = 4*mf*mf/kappa/kappa
    return (-1)**n*BigC(l,mu)*kappa/np.pi*np.sqrt(Nc)

def compute_decay(state_data, mf, kappa, decay='p', Nc=3):
    ''' compute decay constant given a state's eigenvectors as a list of (n, m, l, s1, s2, coeff) '''
    s = 0
    mu = 4*mf*mf/kappa/kappa
    for n, m, l, s1, s2, c in state_data:
        if m == 0 and l % 2 == 0 and s1 + s2 == 0:
            if decay == 'p': 
                s += (-1)**n*BigC(l,mu)*np.sign(s1)*c
            if decay == 'v':
                s += (-1)**n*BigC(l,mu)*c
    return np.abs(s*kappa/np.pi*np.sqrt(Nc))

def compute_decay_base_on_quantum_amp(state_data, quantum_amp, mf, kappa, decay='p', Nc=3):
    ''' compute decay constant given a state's eigenvectors as a list of (n, m, l, s1, s2, coeff) '''
    s = 0
    mu = 4*mf*mf/kappa/kappa
    idx = 0
    for n, m, l, s1, s2, _ in state_data:
        c = quantum_amp[idx]
        if m == 0 and l % 2 == 0 and s1 + s2 == 0:
            if decay == 'p': 
                s += (-1)**n*BigC(l,mu)*np.sign(s1)*c
            if decay == 'v':
                s += (-1)**n*BigC(l,mu)*c
        idx += 1
    return s*kappa/np.pi*np.sqrt(Nc)

# compute parton distribution function
def gfac(x):
    return gamma(x+1)
def gbinom(x, y):
    return gfac(x)/gfac(y)/gfac(x-y)
def JacobiP(n, a, b, x):
    return sum(binom(n+a, n-s)*binom(n+b,s)*((x-1)/2)**s*((x+1)/2)**(n-s) for s in range(0,n+1))

def chi(l, mu, x):
    coeff = np.sqrt((2*l+2*mu+1)*gamma(l+2*mu+1)/gamma(l+mu+1)*gamma(l+1)/gamma(l+mu+1))
    return coeff*(x*(1-x))**(mu/2)*JacobiP(l, mu, mu, 2*x-1)

def compute_pdf(state_data, x, mf, kappa, verbose=0):
    '''state_data is a list of the form (n, m, l, s1, s2, coeff) '''
    mu = 4*mf**2/kappa**2
    groups = defaultdict(list)
    res = 0
    for n, m, l, s1, s2, c in state_data:
        groups[(n,m,s1,s2)].append([n, m, l, s1, s2, c])
    for group_key, group in groups.items():
        if verbose==1: print("Group: (n,m,s1,s2) =", tuple(map(int,group_key)))
        for i in range(len(group)):
            for j in range(len(group)):
                l1, l2 = int(group[i][2]), int(group[j][2])
                c1, c2 = group[i][5], group[j][5]
                if verbose==1: print(f"l1={l1}, l2={l2}, c1*c2={c1*c2:.8f}, val={c1*c2*chi(l1, mu, x)*chi(l2, mu, x):.8}")
                res += c1*c2*chi(l1, mu, x)*chi(l2, mu, x)
    return res

def compute_pdf_on_quantum_amp(state_data, quantum_amp, x, mf, kappa, verbose=0):
    ''' compute pdf given a state's eigenvectors as a list of (n, m, l, s1, s2, coeff) '''
    mu = 4*mf**2/kappa**2
    groups = defaultdict(list)
    res = 0
    idx = 0
    for n, m, l, s1, s2, c in state_data:
        groups[(n,m,s1,s2)].append([n, m, l, s1, s2, c, idx])
        idx += 1
    for group_key, group in groups.items():
        if verbose==1: print("Group: (n,m,s1,s2) =", tuple(map(int,group_key)))
        for i in range(len(group)):
            for j in range(len(group)):
                l1, l2 = int(group[i][2]), int(group[j][2])
                # c1, c2 = group[i][5], group[j][5]
                idx1, idx2 = group[i][-1], group[j][-1]
                amp_prod = quantum_amp[idx1]*quantum_amp[idx2]
                if verbose==1: print(f"l1={l1}, l2={l2}, c1*c2={amp_prod:.8f}, val={amp_prod*chi(l1, mu, x)*chi(l2, mu, x):.8}")
                res += amp_prod*chi(l1, mu, x)*chi(l2, mu, x)
    return res

class BLFQ_file:
    def __init__(self, ham_fn, spect_fn, evect_fn):
        self.folder = ""
        self.ham_fn = ham_fn
        self.spect_fn = spect_fn
        self.evect_fn = evect_fn
        self.mf = self.get_mf()
        self.kap = self.get_kap()
        self.mass_table = {}
        self.decay_table = {}

    def get_Ham(self, abs_threshold=1e-8):
        ham_path = os.path.join(self.folder, self.ham_fn)
        return load_Ham_data_by_threshold(ham_path, abs_threshold=abs_threshold)
    
    def get_mf(self):
        fn = os.path.split(self.spect_fn)[1]
        info_list = fn.split("_")
        return float(info_list[2][2:].replace("p", "."))
        
    def get_kap(self):
        fn = os.path.split(self.spect_fn)[1]
        info_list = fn.split("_")
        return float(info_list[3][3:].replace("p", "."))
    
    def get_mass(self, idx=0):
        '''GeV^2, squared mass'''
        if idx not in self.mass_table:
            masses = load_spectrum(os.path.join(self.folder, self.spect_fn))
            self.mass_table = {i: mass for i, mass in enumerate(masses)}
        return self.mass_table[idx]
    
    def get_decay(self, idx=0, decay='p'):
        '''GeV'''
        if (idx, decay) not in self.decay_table:
            state_data = self.get_evect(idx=idx)
            self.decay_table[(idx, decay)] = compute_decay(state_data, self.mf, self.kap, decay=decay, Nc=3)
        return self.decay_table[(idx, decay)]
    
    def get_delta(self):
        '''GeV^2, sum of mass square abs diff'''
        EXP_MASS_SQUARED = np.array([0.140**2, 0.770**2])
        return np.sum(np.abs(EXP_MASS_SQUARED - np.array([self.get_mass(0), self.get_mass(1)])))
    
    def get_evect(self, idx=0):
        evect_data = load_evect_data(os.path.join(self.folder, self.evect_fn))
        state_data = np.concatenate((evect_data[:,0:5], evect_data[:,5+idx:5+idx+1]), axis=1)
        return state_data
    
    def get_pdf(self, x, idx=0, verbose=0):
        state_data = self.get_evect(idx)
        return compute_pdf(state_data, x, self.mf, self.kap, verbose=verbose)

    def get_all_decays(self, n_state=4):
        ''' compute all decays in MeV'''
        print("#       f_p       f_v")
        for idx in range(n_state):
            print("{}  {:>8.3f}  {:>8.3f}".format(
                idx+1, 1000*self.get_decay(idx=idx, decay='p'), 1000*self.get_decay(idx=idx, decay='v'))
                )
    