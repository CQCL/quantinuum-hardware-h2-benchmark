# Copyright 2023 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

# The even-odd averaged correlation function C_0^xx(r,t) from Eq. (63)
# of https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.094304
def avg_mps_correlation_function(counts, rs, num_iterations, num_burn_in_iterations, num_skip_iterations=1, mode=None):
    corrs       = np.zeros(len(rs))
    num_samples = np.zeros(len(rs))
    
    if mode is None or mode == 'average':
        xs = [0,1]
        ys = [0,1]
    elif mode == 'even':
        xs = [0]
        ys = [0]
    elif mode == 'odd':
        xs = [1]
        ys = [0]
    else:
        raise ValueError(f'Mode {mode} is not supported.')
        
    initial_iteration = 2 * num_burn_in_iterations
    iteration_step    = 2 * num_skip_iterations
    
    for conf_str in counts:
        # Combine the ClassicalRegister bits together and reorder them,
        # so that they go from left to right instead of right to left as stored.
        conf = ''.join(conf_str.split())
        conf = conf[::-1]
        
        for indr in range(len(rs)):
            r = rs[indr]
            # This ensures that there is no even/odd artifacts
            # in the averaged correlation function.
            final_iteration = 2*(num_iterations - (int(np.ceil(r/2))+1))
            for x in xs:
                for y in ys:
                    for j in range(initial_iteration, final_iteration, iteration_step):
                        
                        site1 = j + x
                        site2 = j + x + r + y
                        
                        if site1 < len(conf) and site2 < len(conf):
                            if conf[site1] == conf[site2]:
                                corrs[indr] += 1.0 * counts[conf_str]
                            else:
                                corrs[indr] += -1.0 * counts[conf_str]
                                
                            num_samples[indr] += counts[conf_str]
    
    inds_nonzero_samples = np.where(num_samples > 0)[0]
    corrs[inds_nonzero_samples] /= num_samples[inds_nonzero_samples]
    
    corrs_stderrs = np.zeros(len(rs))
    corrs_stderrs[inds_nonzero_samples] = np.sqrt((1.0 - np.abs(corrs[inds_nonzero_samples])**2.0) / num_samples[inds_nonzero_samples])
    
    return corrs, corrs_stderrs


# Compute the correlation function data from the raw job bitstring data.
def correlation_function(results_dict, rs, ts, num_iterations, num_burn_in_iterations=4):
    num_jobs     = len(results_dict)
    ts_jobs      = []
    results_jobs = []
    for ind_job in range(num_jobs):
        results_jobs.append(results_dict[f"{ind_job}"]['results'])
        ts_jobs.append(results_dict[f"{ind_job}"]['t'])

    num_burn_in_iterations_list = num_burn_in_iterations * np.ones(len(ts), dtype=int) # The # of burn-in iterations for the MPS for each time step
    mode                        = 'average'

    # All of the data.
    corrs_all         = np.zeros((len(rs), len(ts)))
    corrs_all_stderrs = np.zeros((len(rs), len(ts)))

    # The unleaked data.
    corrs         = np.zeros((len(rs), len(ts)))
    corrs_stderrs = np.zeros((len(rs), len(ts)))

    fractions_unleaked = []

    for indt in range(len(ts)):
        t                      = ts[indt]
        num_burn_in_iterations = num_burn_in_iterations_list[indt]
        
        detect_leakage = False # Flag that is triggered if a register called "leak" exists.
        
        # Process the results of all the batches for a single time t
        # into a counts dictionary formatted like qiskit's qasm simulator 
        # counts dictionary.
        all_counts = dict()
        inds_jobs  = np.where(np.array(ts_jobs) == t)[0]
        for ind_job in inds_jobs:
            results     = results_jobs[ind_job]
            num_cregs   = len(results) 
            num_samples = len(results['c0'])
            
            for ind_sample in range(num_samples):
                conf_str_list = []
                for ind_creg in range(num_cregs-1, -1, -1):
                    creg_name = f'c{ind_creg}' 
                    if (ind_creg == num_cregs - 1) and ('leak' in results):
                        creg_name      = f'leak'
                        detect_leakage = True
                    conf_sample_str = results[creg_name][ind_sample]
                    
                    conf_str_list.extend([conf_sample_str[ind_str] for ind_str in range(len(conf_sample_str))])
                    if (ind_creg != 0):
                        conf_str_list.append(' ')
                            
                conf = ''.join(conf_str_list)
                if conf in all_counts:
                    all_counts[conf] += 1
                else:
                    all_counts[conf] = 1

        counts = all_counts
        
        # Keep track of the unleaked data.
        counts_unleaked = dict()
        if detect_leakage:
            for conf in counts:
                if conf[0] == '0':
                    new_conf = conf[2:] # Remove the leakage flag
                    if new_conf not in counts_unleaked:
                        counts_unleaked[new_conf] = counts[conf]
                    else:
                        counts_unleaked[new_conf] += counts[conf]
        
        # The fraction of shots where bond qubit leakage was not measured.
        fraction_unleaked  = np.sum(np.array([counts_unleaked[conf] for conf in counts_unleaked]))
        fraction_unleaked /= np.sum(np.array([counts[conf] for conf in counts]))

        fractions_unleaked.append(fraction_unleaked)
        
        # All data.
        corrs_all[:, indt], corrs_all_stderrs[:, indt] = avg_mps_correlation_function(counts, rs, num_iterations, num_burn_in_iterations, mode=mode) 
        # Unleaked data.
        corrs[:, indt], corrs_stderrs[:, indt] = avg_mps_correlation_function(counts_unleaked, rs, num_iterations, num_burn_in_iterations, mode=mode) 

    return (corrs, corrs_stderrs, corrs_all, corrs_all_stderrs, fractions_unleaked)
