import correlation_toolbox.helper as ch
import correlation_toolbox.correlation_analysis as corr
import json
import numpy as np
import os
import sys

from multiarea_model.multiarea_model import MultiAreaModel

data_path = sys.argv[1]
label = sys.argv[2]
area = sys.argv[3]


load_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'rate_time_series_full')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'synaptic_input')

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    sim_params = json.load(f)
T = sim_params['T']


"""
Create MultiAreaModel instance to have access to data structures
"""
connection_params = {'g': -11.,
                     'cc_weights_factor': sim_params['cc_weights_factor'],
                     'cc_weights_I_factor': sim_params['cc_weights_I_factor'],
                     'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'connection_params': connection_params}
M = MultiAreaModel(network_params)


"""
Synaptic filtering kernel
"""
t = np.arange(0., 20., 1.)
tau_syn = M.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
kernel = np.exp(-t / tau_syn)

    
"""
Load rate time series
"""
rate_time_series = {}
for source_area in M.area_list:
    rate_time_series[source_area] = {}
    for source_pop in M.structure[source_area]:
        fn = os.path.join(load_path,
                          'rate_time_series_full_{}_{}.npy'.format(source_area, source_pop))
        dat = np.load(fn)
        rate_time_series[source_area][source_pop] = dat


synaptic_input_list = []
N_list = []
for pop in M.structure[area]:
    time_series = np.zeros(int((sim_params['T'] - 500.)))
    for source_area in M.area_list:
        for source_pop in M.structure[source_area]:
            weight = M.W[area][pop][source_area][source_pop]
            time_series += (rate_time_series[source_area][source_pop] *
                            abs(weight) *
                            M.K[area][pop][source_area][source_pop])
    syn_current = np.convolve(kernel, time_series, mode='same')
    synaptic_input_list.append(syn_current)
    N_list.append(M.N[area][pop])

    fp = '_'.join(('synaptic_input',
                   area,
                   pop))
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    np.save('{}/{}.npy'.format(save_path, fp), syn_current)

synaptic_input_list = np.array(synaptic_input_list)
area_time_series = np.average(synaptic_input_list, axis=0, weights=N_list)

fp = '_'.join(('synaptic_input',
               area))
np.save('{}/{}.npy'.format(save_path, fp), area_time_series)

par = {'areas': M.area_list,
       'pops': 'complete',
       'resolution': 1.,
       't_min': 500.,
       't_max': T}
fp = '_'.join(('synaptic_input',
               'Parameters.json'))
with open('{}/{}'.format(save_path, fp), 'w') as f:
    json.dump(par, f)
