import json
import numpy as np
import os

import correlation_toolbox.helper as ch
from multiarea_model import MultiAreaModel
import sys

"""
Compute correlation coefficients for a subsample
of neurons for the entire network from raw spike files of a given simulation.
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

tmin = 500.
subsample = 2000
resolution = 1.

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

spike_data = {}
cc_dict = {}
for area in M.area_list:
    cc_dict[area] = {}
    LvR_list = []
    N = []
    for pop in M.structure[area]:
        fp = '-'.join((label,
                       'spikes',  # assumes that the default label for spike files was used
                       area,
                       pop))
        fn = '{}/{}.npy'.format(load_path, fp)
        # +1000 to ensure that we really have subsample non-silent neurons in the end
        spikes = np.load(fn)
        ids = np.unique(spikes[:, 0])
        dat = ch.sort_gdf_by_id(spikes, idmin=ids[0], idmax=ids[0]+subsample+1000)
        bins, hist = ch.instantaneous_spike_count(dat[1], resolution, tmin=tmin, tmax=T)
        rates = ch.strip_binned_spiketrains(hist)[:subsample]
        cc = np.corrcoef(rates)
        cc = np.extract(1-np.eye(cc[0].size), cc)
        cc[np.where(np.isnan(cc))] = 0.
        cc_dict[area][pop] = np.mean(cc)
    
fn = os.path.join(save_path,
                  'corrcoeff.json')
with open(fn, 'w') as f:
    json.dump(cc_dict, f)
