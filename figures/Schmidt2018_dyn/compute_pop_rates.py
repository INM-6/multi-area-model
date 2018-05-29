import json
import numpy as np
import os

from multiarea_model.analysis_helpers import pop_rate
from multiarea_model import MultiAreaModel
import sys

"""
Compute stationary spike rates for the entire network from raw spike
files of a given simulation.
"""

data_path = sys.argv[1]
label = sys.argv[2]

load_path = os.path.join(data_path,
                         label,
                         'recordings')
save_path = os.path.join(data_path,
                         label,
                         'Analysis')

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    sim_params = json.load(f)
T = sim_params['T']

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

spike_data = {}
pop_rates = {}
for area in M.area_list:
    pop_rates[area] = {}
    rate_list = []
    N = []
    for pop in M.structure[area]:
        fp = '-'.join((label,
                       'spikes',  # assumes that the default label for spike files was used
                       area,
                       pop))
        fn = '{}/{}.npy'.format(load_path, fp)
        dat = np.load(fn)
        print(area, pop)
        pop_rates[area][pop] = pop_rate(dat, 500., T, M.N[area][pop])
        rate_list.append(pop_rates[area][pop])
        N.append(M.N[area][pop])
    pop_rates[area]['total'] = np.average(rate_list, weights=N)
    
fn = os.path.join(save_path,
                  'pop_rates.json')
with open(fn, 'w') as f:
    json.dump(pop_rates, f)
