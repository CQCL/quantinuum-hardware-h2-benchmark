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

''' Functions for reporting SPAM data from Quantinuum. '''

import pandas as pd
import numpy as np
from util import avg_uncertainty

from loading_functions import load_data
from zone_names import *


def report(experiment: str):
    ''' Returns DataFrame containing summary of results. '''

    data = load_data(experiment)
    spam_results = data['survival']

    spam_results = {
        zone_labels_3[key]: {
            s: f/data['shots'] 
            for s, f in fid.items()
        }
        for key, fid in spam_results.items()
    }

    df1 = pd.DataFrame.from_dict(spam_results).transpose()
    df1.rename(columns={'0': '0 SPAM error', '1': '1 SPAM error'}, inplace=True)

    avg = {
        q: {
            'Avg. SPAM error': (res['0'] + res['1'])/2,
            'Avg. SPAM error uncertainty': np.sqrt(res['0']*(1 - res['0']) + res['1']*(1 - res['1']))/2/np.sqrt(data['shots'])
        }
        for q, res in spam_results.items()
    }
    df2 = pd.DataFrame.from_dict(avg).transpose()
    
    result = pd.concat([df1, df2], axis=1).reindex(df1.index)
    result.rename(columns={result.columns[0]: 'Qubits'})
    result.loc['Mean'] = result.mean()

    # change uncertainties to geometric means
    result['Avg. SPAM error uncertainty']['Mean'] = avg_uncertainty(
        result['Avg. SPAM error uncertainty'].head(len(result['Avg. SPAM error uncertainty']) - 1).to_list()
    )

    result['Avg. SPAM error'] = result['Avg. SPAM error'].map(lambda x: 1 - x)
    result['0 SPAM error'] = result['0 SPAM error'].map(lambda x: 1 - x)
    result['1 SPAM error'] = result['1 SPAM error'].map(lambda x: 1 - x)
    pd.set_option('display.float_format', lambda x: '%.3E' % x)

    result = result[['Avg. SPAM error', 'Avg. SPAM error uncertainty', '0 SPAM error', '1 SPAM error']]

    return result

def spam_combined(experiment: str):

    data = load_data(experiment)
    spam_results = data['survival']

    nqubits = len(spam_results)
    res = {i: 
        sum(
            spam_results[q][i]/data['shots']/nqubits
            for q in spam_results
        )
        for i in ['0', '1']
    }
    fid = (res['0'] + res['1'])/2
    unc = np.sqrt(res['0']*(1 - res['0']) + res['1']*(1 - res['1']))/2/np.sqrt(data['shots']*nqubits)
 
    return 1 - fid, unc