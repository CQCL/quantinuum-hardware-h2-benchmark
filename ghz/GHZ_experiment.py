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
import matplotlib.pyplot as plt

# functions for analyzing GHZ data

def retrieve(GHZ_dict):

    hists = {}

    for setting in GHZ_dict['results']:
        results = GHZ_dict['results'][setting]
        outcomes = {}
        for b in results['c']:
            if b in outcomes:
                outcomes[b] += 1
            else:
                outcomes[b] = 1

        hists[setting] = outcomes
        
    return hists


def exp_value(outcomes):
    
    exp_val = 0.0
    shots = sum(outcomes.values())
    for outcome in outcomes:
        parity = (-1)**(outcome.count('1'))
        counts = outcomes[outcome]
        exp_val += parity*counts/shots
    
    return exp_val


def fidelity(hists):
    
    # read in number of qubits and shots
    n = len(list(list(hists.values())[0].keys())[0])
    #shots = sum(list(hists.values())[0].values())
    N = len(hists)
        
    f = 0.0 # fidelity estimate
    var = 0.0 # variance of fidelity estimate
    
    for name in hists:
        #meas_basis = name[1]
        outcomes = hists[name]
        shots = sum(outcomes.values())
            
        if 'Z' in name:
            p = 0.0 # success prob
            if '0'*n in outcomes:
                p0 = outcomes['0'*n]/shots
                p += p0
            if '1'*n in outcomes:
                p1 = outcomes['1'*n]/shots
                p += p1
            f += p/N
            var += (p*(1-p))
            
        
        # for phase scan method
        elif 'theta' in name:
            index = name.find('a')+1
            k = int(name[index:len(name)-2])
            sign = (-1)**k
            # compute expectation value of spin operator
            exp_val = exp_value(outcomes)
            # update fidelity estimate and variance
            f += exp_val*sign/N
            var += (1+exp_val)*(1-exp_val)
    
    var = var/(shots*N**2)
    if np.abs(var) < 10**(-12): # handle possible rounding error
        var = 0.0
        
    std = np.sqrt(var)
    
    return f, std
    
    
def merge_outcomes(out1, out2):
    """ combine outcomes from different circuit executions (useful for RC)
        out1, out2 (dict)
    """
    
    outcomes = out1
    for b_str in out2:
        counts = out2[b_str]
        if b_str in outcomes:
            outcomes[b_str] += counts
        elif b_str not in outcomes:
            outcomes[b_str] = counts
    
    return outcomes


# functions for making plots of GHZ population and parities

def plot_populations(exp_list, **kwargs):
        
    ylim = kwargs.get('ylim', (0,0.55))
    
    colors = [plt.get_cmap("tab10").colors[i] for i in range(4)]
    labels = ['N=20', 'N=26', 'N=32', 'N=32 adap.']
    n_qubits = [20, 26, 32, 32]
    
    w = 0.2
    for i, exp in enumerate(exp_list):
        hists = retrieve(exp)
        
        co = colors[i]

        # read in number of qubits and shots
        n = n_qubits[i]
        pops = {}
        for setting in hists:
            if 'Z' in setting:
                outcomes = hists[setting]
                pops = merge_outcomes(pops, outcomes)

        shots = sum(pops.values())
        if n <= 12:
            x = ['0'*n, 'Everything else', '1'*n]
        elif n > 12:
            x = ['00...0', 'Everything else', '11...1']

        y0 = pops['0'*n]/shots
        y1 = pops['1'*n]/shots
        y_else = 1-y0-y1

        y = [y0, y_else, y1]
        yerr = [np.sqrt(p*(1-p)/shots) for p in y]

        plt.bar(np.array(range(3))-(2-i)*w+0.1,y, width=w, yerr=yerr, color=co, label=labels[i],
                alpha=0.5, capsize=5)
    
    plt.ylim(ylim)
    plt.xticks(ticks=[0,1,2], labels=x, fontsize=15)
    plt.yticks(fontsize=12)
    plt.ylabel('Probability', fontsize=15)
    plt.legend(fontsize=11)
    plt.text(-0.875, 0.575, '(a)', size=16)
    #plt.savefig('GHZ_populations.pdf', format='pdf')
    plt.show()
        
        
def plot_parities(exp_list):
    
    colors=[plt.get_cmap("tab10").colors[i] for i in range(4)]
    labels = ['N=20', 'N=26', 'N=32', 'N=32 adap.']
    n_qubits = [20, 26, 32, 32]
    
    for exp_num, exp in enumerate(exp_list):
        hists = retrieve(exp)
        
        co = colors[exp_num]

        # read in n
        n = n_qubits[exp_num]

        parities = []
        stds = []

        for k in range(1,n+1):
            for name in hists:
                if 'theta' + str(k) in name and name[name.find('a')+len(str(k))+2] == ')':
                    outcomes = hists[name]
                    shots = sum(outcomes.values())
                    exp_val = exp_value(outcomes)
                    p = (exp_val + 1)/2
                    var = 4*p*(1-p)/shots
                    parities.append(exp_val)
                    stds.append(np.sqrt(var))

        # make parity plot
        x = [i/n for i in range(1,n+1)]
        y_avg1 = np.mean([parities[i] for i in [2*j for j in range(int(n/2))]])
        y_avg2 = np.mean([parities[i] for i in [2*j+1 for j in range(int(n/2))]])

        plt.errorbar(x, parities, yerr=stds, fmt='o', color=co, label=labels[exp_num], alpha=0.5,
                    markersize=3, capsize=5)
        plt.axhline(y=y_avg1, color=co, linestyle='--')
        plt.axhline(y=y_avg2, color=co, linestyle='--')
    
    plt.xlabel('Measurement angle (units of pi)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Parity', fontsize=15)
    plt.ylim((-1.1,1.1))
    plt.legend(fontsize=11)
    plt.text(-0.2, 1.2, '(b)', size=16)
    plt.tight_layout()
    #plt.savefig('GHZ_parity.pdf', format='pdf')
    plt.show()

