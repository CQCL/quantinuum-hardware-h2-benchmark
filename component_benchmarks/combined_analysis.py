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

import pandas as pd

from decay_analysis_functions import decay_analysis_combined
from rb_analysis_functions import rb_analysis_combined
from spam_reporting_functions import spam_combined


def combined_report(test_list: list):
    ''' Report estimates from all methods. '''

    renamed = {
        '1Q_RB': 'Single-qubit gate error',
        '2Q_RB': 'Two-qubit gate error',
        '1Q_RB_SE': 'Single-qubit leakage',
        '2Q_RB_SE': 'Two-qubit leakage',
        'SU4_RB': 'SU(4) gate error',
        'Transport_1QRB': 'Memory error',
        'Measurement_crosstalk': 'Measurement crosstalk error',
        'Reset_crosstalk': 'Reset crosstalk error',
        'SPAM': 'SPAM error'
    }

    df = {}
    for test in test_list:
        df[renamed[test]] = [None, None]
        if 'RB' in test:
            df[renamed[test]][0], df[renamed[test]][1] = rb_analysis_combined(test)
            try:
                tmp0, tmp1 = rb_analysis_combined(
                    test, 
                    'leakage_postselect'
                )
                df[renamed[test+'_SE']] = [tmp0, tmp1]
            except KeyError:
                pass
        elif test =='Measurement_crosstalk' or test == 'Reset_crosstalk':
            df[renamed[test]][0], df[renamed[test]][1] = decay_analysis_combined(test)
        elif test == 'SPAM':
            df[renamed[test]][0], df[renamed[test]][1] = spam_combined(test)

    result = pd.DataFrame.from_dict(df).transpose()
    result.rename(columns={0: 'Magnitude', 1: 'Uncertainty'}, inplace=True)
    pd.set_option('display.float_format', lambda x: '%.3E' % x)

    return result