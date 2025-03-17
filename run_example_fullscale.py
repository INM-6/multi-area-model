import numpy as np
import os

from multiarea_model import MultiAreaModel
from start_jobs import start_job
from config import submit_cmd, jobscript_template
from config import base_path

"""
Example script showing how to simulate the multi-area model
on a cluster.

We choose the same configuration as in
Fig. 3 of Schmidt et al. (2018).

"""

"""
Full model. Needs to be simulated with sufficient
resources, for instance on a compute cluster.
"""
d = {}

conn_params = {'cc_weights_factor': 1.0, # run model in Ground State
               'cc_weights_I_factor': 1.0}

network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params}

sim_params = {'t_sim': 2000.,
              'num_processes': 720,
              'local_num_threads': 1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True)
p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
start_job(M.simulation.label, submit_cmd, jobscript_template)
