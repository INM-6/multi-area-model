import json
import neo
import numpy as np
import os
import quantities as pq

from multiarea_model.analysis_helpers import pop_rate_time_series
from elephant.statistics import instantaneous_rate
from multiarea_model import MultiAreaModel
import sys

"""
Compute time series of population-averaged spike rates for a given
area from raw spike files of a given simulation.

Implements three different methods:
- binned spike histograms on all neurons ('full')
- binned spike histograms on a subsample of 140 neurons ('subsample')
- spike histograms convolved with a Gaussian kernel of optimal width
  after Shimazaki et al. (2010)
"""

assert(len(sys.argv) == 5)

data_path = sys.argv[1]
label = sys.argv[2]
area = sys.argv[3]
method = sys.argv[4]

assert(method in ['subsample', 'full', 'auto_kernel'])
# subsample : subsample spike data to 140 neurons to match the Chu 2014 data
# full : use spikes of all neurons and compute spike histogram with bin size 1 ms
# auto_kernel : use spikes of all neurons and convolve with Gaussian
#               kernel of optimal width using the method of Shimazaki et al. (2012)
#               (see Method parts of the paper)

load_path = os.path.join(data_path,
                         label,
                         'recordings')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'rate_time_series_{}'.format(method))
try:
    os.mkdir(save_path)
except FileExistsError:
    pass

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    sim_params = json.load(f)
T = sim_params['T']

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

time_series_list = []
N_list = []
for pop in M.structure[area]:
    fp = '-'.join((label,
                   'spikes',  # assumes that the default label for spike files was used
                   area,
                   pop))
    fn = '{}/{}.npy'.format(load_path, fp)
    spike_data = np.load(fn)
    spike_data = spike_data[np.logical_and(spike_data[:, 1] > 500.,
                                           spike_data[:, 1] <= T)]
    if method == 'subsample':
        all_gid = np.unique(spike_data[:, 0])
        N = int(np.round(140 * M.N[area][pop] / M.N[area]['total']))

        i = 0
        s = 0
        gid_list = []
        while s < N:
            rate = spike_data[:, 1][spike_data[:, 0] == all_gid[i]].size / (1e-3 * (T - 500.))
            if rate > 0.56:
                gid_list.append(all_gid[i])
                s += 1
            i += 1
        spike_data = spike_data[np.isin(spike_data[:, 0], gid_list)]
        kernel = 'binned_subsample'
    if method == 'full':
        N = M.N[area][pop]  # Assumes that all neurons were recorded
        kernel = 'binned'
        
    if method in ['subsample', 'full']:
        time_series = pop_rate_time_series(spike_data, N, 500., T,
                                           resolution=1.)

    if method == 'auto_kernel':
        # To reduce the computational load, the time series is only computed until 10500. ms
        T = 10500.
        N = M.N[area][pop]  # Assumes that all neurons were recorded
        st = neo.SpikeTrain(spike_data[:, 1] * pq.ms, t_stop=T*pq.ms)
        time_series = instantaneous_rate(st, 1.*pq.ms, t_start=500.*pq.ms, t_stop=T*pq.ms)
        time_series = np.array(time_series)[:, 0] / N

        kernel = 'auto'
        
    time_series_list.append(time_series)
    N_list.append(N)
    
    fp = '_'.join(('rate_time_series',
                   method,
                   area,
                   pop))
    np.save('{}/{}.npy'.format(save_path, fp), time_series)

time_series_list = np.array(time_series_list)
area_time_series = np.average(time_series_list, axis=0, weights=N_list)

fp = '_'.join(('rate_time_series',
               method,
               area))
np.save('{}/{}.npy'.format(save_path, fp), area_time_series)

par = {'areas': M.area_list,
       'pops': 'complete',
       'kernel': kernel,
       'resolution': 1.,
       't_min': 500.,
       't_max': T}
fp = '_'.join(('rate_time_series',
               method,
               'Parameters.json'))
with open('{}/{}'.format(save_path, fp), 'w') as f:
    json.dump(par, f)
