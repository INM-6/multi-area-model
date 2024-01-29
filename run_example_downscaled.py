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
               'cc_weights_factor': 1.0, # run model in Ground State
               'cc_weights_I_factor': 1.0}
network_params = {'N_scaling': 0.01,
                  'K_scaling': 0.01,
                  'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates.json')}

sim_params = {'t_sim': 2000.,
              'num_processes': 1,
              'local_num_threads': 1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True)

p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
M.simulation.simulate()
