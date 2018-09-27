import numpy as np
import os

from multiarea_model import MultiAreaModel
from config import base_path

"""
Down-scaled model.
Neurons and indegrees are both scaled down to 10 %.
Can usually be simulated on a local machine.

Warning: This will not yield reasonable dynamical results from the
network and is only meant to demonstrate the simulation workflow.
"""
d = {}
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
                  'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates.json'),
                  'input_params': input_params,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 2000.,
              'num_processes': 1,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
M.simulation.simulate()
