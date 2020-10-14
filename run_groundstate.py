import os

import multiarea_model as model
from config import base_path
import time
import nest


time_start = time.time()

conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
               'g': -11.,
               'K_stable': 'K_stable.npy',
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': 0.01,
                  'K_scaling': 0.01,
                  'fullscale_rates': os.path.join(
                      base_path,
                      'tests/fullscale_rates.json'
                      ),
                  'input_params': input_params,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 2000.,
              'num_processes': 1,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

parameters = model.get_default_parameters()
new_params = {
        'sim_dict': sim_params,
        'network_dict': network_params
        }
model.nested_update(parameters, new_params)

M = model.Model(parameters)

time_model = time.time()

M.create()

time_create = time.time()

M.connect()

time_connect = time.time()

M.simulate(parameters['sim_dict']['t_sim'])

time_simulate = time.time()

print(
    '\nTimes of Rank {}:\n'.format(
        nest.Rank()) +
    '  Total time:          {:.3f} s\n'.format(
        time_simulate -
        time_start) +
    '  Time to initialize:  {:.3f} s\n'.format(
        time_model -
        time_start) +
    '  Time to create:      {:.3f} s\n'.format(
        time_create -
        time_model) +
    '  Time to connect:     {:.3f} s\n'.format(
        time_connect -
        time_create) +
    '  Time to simulate:    {:.3f} s\n'.format(
        time_simulate -
        time_create))
