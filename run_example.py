from multiarea_model import MultiAreaModel
from start_jobs import start_job
from config import submit_cmd, jobscript_template

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
conn_params = {'g': -11.,
               'K_stable': 'K_stable.npy',
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 2000.,
              'num_processes': 720,
              'num_rec_processes': 1,
              'local_num_threads': 1,
              'input_params': input_params,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r = M.theory.integrate_siegert()
start_job(M.simulation.label, submit_cmd, jobscript_template)


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
                  'fullscale_rates': 'tests/fullscale_rates.json',
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 2000.,
              'num_processes': 1,
              'local_num_threads': 1,
              'input_params': input_params,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r = M.theory.integrate_siegert()
M.simulation.simulate()
