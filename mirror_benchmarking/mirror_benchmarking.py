# Copyright 2023 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from scipy.optimize import curve_fit

# functions for analyzing mirror benchmarking data

def retrieve(MB_dict):

    hists = {}

    for setting in MB_dict['results']:
        results = MB_dict['results'][setting]
        outcomes = {}
        for b in results['c']:
            if b in outcomes:
                outcomes[b] += 1
            else:
                outcomes[b] = 1

        hists[setting] = outcomes
        
    return hists


def analyze_hists(hists, surv_state, error_bars=True, shots=100):
        
        
        n = len(list(list(hists.values())[0].keys())[0])
        
        success_probs = get_success_probs(hists, surv_state)
        avg_success_probs = get_avg_success_probs(success_probs)
        seq_len = list(avg_success_probs.keys())
        seq_len.sort()
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        # estimate unitarity and TQ gate fidelity
        x = list(seq_len)
        y = [avg_success_probs[L] for L in x]
        # perform best fit
        popt, pcov = curve_fit(fit_func, x, y, p0=[0.9, 0.9], bounds=(0,1))
        unitarity = popt[1]
        f_avg = unitarity2TQ_fidelity(unitarity, n)
        
        # bootstrap for error bars
        if error_bars == True:
            error_data = compute_error_bars(success_probs, shots, n)
        else:
            error_data=None
            
        return {'f_avg':f_avg, 'unitarity':unitarity, 'avg_success_probs':avg_success_probs, 'error_data':error_data}


def get_success_probs(hists, surv_state):
    
    success_probs = {}

    for setting in hists:
        i0, i1 = setting.find('('), setting.find(',')
        L = int(setting[i0+1:i1])
        outcomes = hists[setting]
        exp_out = surv_state[setting]
        p = success_probability(exp_out, outcomes)
        if L in success_probs:
            success_probs[L].append(p)
        else:
            success_probs[L] = [p]
            
    seq_len = list(success_probs.keys())
    seq_len.sort()
    success_probs = {L:success_probs[L] for L in seq_len}

    return success_probs


def get_avg_success_probs(success_probs):

    avg_success_probs = {}
    for L in success_probs:
        avg_success_probs[L] = np.mean(success_probs[L])
        
    seq_len = list(avg_success_probs.keys())
    seq_len.sort()
    avg_success_probs = {L:avg_success_probs[L] for L in seq_len}

    return avg_success_probs

def success_probability(exp_out, outcomes):
    
    shots = sum(list(outcomes.values()))
    
    if exp_out in outcomes:
        p = outcomes[exp_out]/shots
    else:
        p = 0.0
    
    return p


def true_unitarity(TQ_err, n):
    """ TQ_err: depolarizing parameter
             n: number of qubits (must be even)
    """
    
    from scipy.special import comb
    
    d = 2**n
    n_pairs = int(n/2) # number of qubit pairs
    
    # unitarity
    u = 0.0
    
    # sum over weights of Paulis
    for w in range(1,n_pairs+1):
        c1 = comb(n_pairs, w, exact=True)
        S = 0.0
        for j in range(n_pairs-w+1):
            c2 = comb(n_pairs-w, j, exact=True)
            S += c2*(TQ_err**j)*(1-TQ_err)**(n_pairs-j)
        u += (15**w)*c1*S**2/(d**2-1)
    
    return u


def unitarity2TQ_fidelity(u, n, lay_depth=1):
    """ TQ fidelity assuming TQ depolarizing error
        u : unitarity
        n : number of qubits (must be even)
        
        returns F_avg : average fidelity
    """
    
    # first, find TQ depolarizing parameter that gives correct unitarity
    tol = 10**(-6)
    finished = False
    
    # initialize lower and upper limits on depolarizing parameter
    p_0, p_1 = 0.0, 1.0
    
    while finished == False:
        p = (p_0+p_1)/2
        u_true = true_unitarity(p, n)
        if abs(u_true-u) < tol:
            finished = True
        else:
            if u_true > u:
                p_0 = p
            if u_true < u:
                p_1 = p
        #print(p) # for debugging
    
    # convert to TQ average fidelity
    #rescale according to layer depth
    p_rs = 1 - ((1-p)**(1/lay_depth))
    F_avg = 1-3*p_rs/4
    
    return F_avg


def compute_error_bars(success_probs, shots, n):
        
        #n = len(list(list(hists.values())[0].keys())[0])
        seq_len = list(success_probs.keys())
        seq_len.sort()
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        boot_probs = bootstrap(success_probs, shots=shots)
        stds = {}
        for L in seq_len:
            stds[L] = np.std([b_prob[L] for b_prob in boot_probs])
        avg_success_probs_stds = stds   
        boot_unitarity = []
        x = seq_len
        for b_prob in boot_probs:
            b_y = [b_prob[L] for L in x]
            # best fit the bootstrapped success probabilities
            b_popt, b_pcov = curve_fit(fit_func, x, b_y, p0=[0.9, 0.9], bounds=(0,1))
            boot_unitarity.append(b_popt[1])
        
        unitarity_std = np.std(boot_unitarity)
        
        # estimate F_avg_std
        f_avg_std = np.std([unitarity2TQ_fidelity(u, n) for u in boot_unitarity])
        
        error_data = {'avg_success_probs_stds':avg_success_probs_stds, 'unitarity_std':unitarity_std, 'f_avg_std':f_avg_std}
        
        return error_data


def bootstrap(success_probs, num_resamples=100, shots=100):
    """ succ_probs (dict): keys are sequence lengths,
                           values are lists of circuit success probs
    """
    
    # read in sequence lengths
    seq_len = list(success_probs.keys())
    seq_len.sort()
    
    boot_probs = []
    
    for samp in range(num_resamples):
        
        b_succ_probs = {}
        for L in seq_len:
            probs = success_probs[L]
            # non-parametric resample from circuits
            re_probs = np.random.choice(probs, size=len(probs))
            # parametric sample from success probs
            re_probs = np.random.binomial(shots, re_probs)/shots
            b_succ_probs[L] = re_probs
        
        # take average success probs
        b_avg_succ_probs = {L:np.mean(b_succ_probs[L]) for L in seq_len}
        
        boot_probs.append(b_avg_succ_probs)
    

    return boot_probs


def resample_outcomes(outcomes):
    
    shots = sum(list(outcomes.values()))
    b_strs = list(outcomes.keys())
    p = np.array(list(outcomes.values()))/shots # probability distribution
    r = list(np.random.choice(b_strs, size=shots, p=p))
    re_out = {b_str:r.count(b_str) for b_str in set(r)}
    
    return re_out

