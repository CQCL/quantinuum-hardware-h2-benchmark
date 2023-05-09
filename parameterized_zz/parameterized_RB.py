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

# functions for analyzing TQ parameterized RB data

def retrieve(TQRB_dict):

    hists = {}

    for setting in TQRB_dict['results']:
        results = TQRB_dict['results'][setting]
        outcomes = {}
        for b in results['c']:
            if b in outcomes:
                outcomes[b] += 1
            else:
                outcomes[b] = 1

        hists[setting] = outcomes
        
    return hists


def analyze_hists(hists, error_bars=True, **kwargs):
        
    fit_method = kwargs.get('fit_method', 1)
    
    qubits = [(0,1), (2,3), (4,5), (6,7)]

    marginal_hists = marginalize_hists(qubits, hists)
    success_probs = [get_success_probs(hists) for hists in marginal_hists]
    avg_success_probs = [get_avg_success_probs(hists) for hists in marginal_hists]
    f_avg = [estimate_fidelity(avg_succ_probs, fit_method=fit_method) for avg_succ_probs in avg_success_probs]

    # compute error bars
    if error_bars == True:
        error_data = [compute_error_bars(hists, fit_method=fit_method) for hists in marginal_hists]
        f_avg_std = [data['avg_fid_std'] for data in error_data]
    else:
        error_data = None
        f_avg_std = None
        
    return {'avg_success_probs':avg_success_probs, 'f_avg':f_avg, 'error_data':error_data}


def marginalize_hists(qubits, hists):
    """ return list of hists of same length as number of qubit pairs """
    
    if type(qubits[0]) == int:
        qubits = [qubits]
    
    # number of qubits needed
    n = max([max(q_pair) for q_pair in qubits]) + 1
    
    mar_hists = []
    for q_pair in qubits:
        q0, q1 = q_pair
        hists_q = {}
        for name in hists:
            out = hists[name]
            mar_out = {}
            for b_str in out:
                counts = out[b_str]
                # marginalize bitstring
                mar_b_str = b_str[n-1-q1] + b_str[n-1-q0]
                if mar_b_str in mar_out:
                    mar_out[mar_b_str] += counts
                elif mar_b_str not in mar_out:
                    mar_out[mar_b_str] = counts
            # append marginalized outcomes to hists
            hists_q[name] = mar_out
        mar_hists.append(hists_q)
    
    return mar_hists


def success_probability(exp_out, outcomes):
    
    shots = sum(list(outcomes.values()))
    
    if exp_out in outcomes:
        p = outcomes[exp_out]/shots
    else:
        p = 0.0
    
    return p


def get_success_probs(hists):
    
    success_probs = {}
    
    for setting in hists:
        L = int(setting[setting.find('(')+1:setting.find(',')])
        exp_out = setting[-4:-2]
        outcomes = hists[setting]
        p = success_probability(exp_out, outcomes)
        if L in success_probs:
            success_probs[L].append(p)
        else:
            success_probs[L] = [p]
    
    # sort
    seq_len = list(success_probs.keys())
    seq_len.sort()
    success_probs = {L:success_probs[L] for L in seq_len}
    
    
    return success_probs


def get_avg_success_probs(hists):
    
    success_probs = get_success_probs(hists)
    
    avg_success_probs = {}
    for L in success_probs:
        avg_success_probs[L] = np.mean(success_probs[L])
    
    return avg_success_probs


def estimate_fidelity(avg_success_probs, fit_method=1):
    
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L + 1/4
    
    def fit_func2(L, a, f, b):
        return a*f**L + b*(1-2*(1-f))**L + 1/4
    
    
    x = [L for L in avg_success_probs]
    x.sort()
    
        
    y = [avg_success_probs[L] for L in x]
    
    # perform best fit
    if fit_method == 1:
        popt, pcov = curve_fit(fit_func, x, y, p0=[0.75, 0.9], bounds=([0,0], [1,1]))
        avg_fidelity = 1 - 3*(1-popt[1])/4
        
    elif fit_method == 2:
        popt, pcov = curve_fit(fit_func2, x, y, p0=[0.4, 0.8, 0.2], bounds=([0,0,0],[0.5, 1, 0.25]))
        avg_fidelity = (6*popt[1]-1)/5
    
    
    return avg_fidelity


def compute_error_bars(hists, fit_method=1):
    
    
    boot_hists = bootstrap(hists)
    boot_avg_succ_probs = [get_avg_success_probs(b_h) for b_h in boot_hists]
    boot_avg_fids = [estimate_fidelity(avg_succ_prob, fit_method=fit_method)
                     for avg_succ_prob in boot_avg_succ_probs]
    
    
    # read in seq_len and list of Paulis
    seq_len = list(boot_avg_succ_probs[0].keys())
    seq_len.sort()
    
    # process bootstrapped data
    probs_stds = {}
    for L in seq_len:
        probs_stds[L] = np.std([b_p[L] for b_p in boot_avg_succ_probs])
    
    avg_fid_std = np.std([f for f in boot_avg_fids])
    error_data = {'success_probs_stds':probs_stds,
                  'avg_fid_std':avg_fid_std}
    
    return error_data


def bootstrap(hists, num_resamples=100):
    """ non-parametric resampling from circuits
        parametric resampling from hists
    """
    
    # read in seq_len and input states
    seq_len = [4, 50, 100]
    
    boot_hists = []
    for i in range(num_resamples):
        
        # first do non-parametric resampling
        hists_resamp = {}
        for L in seq_len:
            # make list of exp names to resample from
            circ_list = []
            for name in hists:
                if str(L) in name:
                    circ_list.append(name)
            # resample from circ_list
            seq_reps = len(circ_list)
            resamp_circs = np.random.choice(seq_reps, size=seq_reps)
            for s, s2 in enumerate(resamp_circs):
                circ = circ_list[s2]
                name_resamp = str((L, s, circ[-4:-2]))
                outcomes = hists[circ]
                hists_resamp[name_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(resample_hists(hists_resamp))
    
    return boot_hists

def resample_hists(hists):
    """ hists (dict): outcome dictionaries for each circuit label
        returns re_hists (dict): same format, resampled
    """
    
    re_hists = {}
    
    for name in hists:
        outcomes = hists[name]
        re_out = resample_outcomes(outcomes)
        re_hists[name] = re_out
    
    return re_hists

def resample_outcomes(outcomes):
    
    # read in number of shots
    shots = sum(outcomes.values())
    out_list = list(outcomes.keys())
    
    # define probability distribution to resample from
    p = [outcomes[b_str]/shots for b_str in out_list]
    
    # resample
    r = list(np.random.choice(out_list, size=shots, p=p))
    re_out = {b_str:r.count(b_str) for b_str in out_list}
    
    return re_out