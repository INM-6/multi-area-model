import json
import numpy as np
import os
import sys

from multiarea_model import MultiAreaModel
from multiarea_model.analysis_helpers import pop_rate_distribution


"""
Compute histogram of spike rates over single neurons for a given area
from raw spike files of a given simulation.
"""

assert(len(sys.argv) == 4)
data_path = sys.argv[1]
label = sys.argv[2]
area = sys.argv[3]

load_path = os.path.join(data_path,
                         label,
                         'recordings')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'rate_histogram')

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    par = json.load(f)
T = par['T']

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})


# spike data
spike_data = {}
for pop in M.structure[area]:
    fp = '-'.join((label,
                   'spikes',  # assumes that the default label for spike files was used
                   area,
                   pop))
    fn = '{}/{}.npy'.format(load_path, fp)
    spike_data[pop] = np.load(fn)

total_spikes = np.zeros((0, 2))
for pop in M.structure[area]:
    total_spikes = np.vstack((total_spikes,
                              spike_data[pop]))
bins, vals, mean, std = pop_rate_distribution(total_spikes,
                                              500.,
                                              T,
                                              int(np.ceil(
                                                  M.N[area]['total'])))

fn = os.path.join(data_path, label,
                  'Analysis',
                  'rate_histogram',
                  'rate_histogram_{}.npy'.format('bins'))
np.save(fn, bins)

fn = os.path.join(data_path, label,
                  'Analysis',
                  'rate_histogram',
                  'rate_histogram_{}.npy'.format(area))
np.save(fn, vals)


par = {'areas': M.area_list,
       'pops': 'complete',
       'resolution': 1.,
       't_min': 500.,
       't_max': T}
fp = '_'.join(('rate_histogram',
               'Parameters.json'))
with open('{}/{}'.format(save_path, fp), 'w') as f:
    json.dump(par, f)

