import os

from multiarea_model import MultiAreaModel
from start_jobs import start_job
from config import submit_cmd, jobscript_template
from config import base_path

"""
This script provides the code to execute all simulations presented in
Schmidt M, Bakker R, Shen K, Bezgin B, Diesmann M & van Albada SJ
(2018) A multi-scale layer-resolved spiking network model of
resting-state dynamics in macaque cortex.

Needs to be simulated with sufficient
resources, for instance on a compute cluster.
"""

# Common parameter settings
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
sim_params = {'num_processes': 20,  # Needs to be adapted to the HPC system used
              'local_num_threads': 24,  # Needs to be adapted to the HPC system used
              'recording_dict': {'record_vm': False}}


"""
Simulation with  kappa = 1. leading to the low-activity fixed point
shown in Fig. 2.
"""
d = {}

sim_params.update({'t_sim': 10500.})
conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.,
               'fac_nu_ext_6E': 1.,
               'av_indegree_V1': 3950.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params,
                  'input_params': input_params}

M_LA = MultiAreaModel(network_params, simulation=True,
                      sim_spec=sim_params)

# start_job(M_LA.simulation.label, submit_cmd, jobscript_template)


"""
Simulation with kappa = 1.125 leading to the high-activity fixed point
shown in Fig. 2.
"""
sim_params.update({'t_sim': 10500.})
conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

M_HA = MultiAreaModel(network_params, simulation=True,
                      sim_spec=sim_params)
p, r_HA = M_HA.theory.integrate_siegert()
# start_job(M_HA.simulation.label, submit_cmd, jobscript_template)


"""
Simulation with kappa = 1.125, chi=1
Presented in Fig. 2, 4, and 8.
"""
sim_params.update({'t_sim': 10500.})
conn_params = {'g': -11.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2018/K_prime_original.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

M_stab = MultiAreaModel(network_params, simulation=True,
                        sim_spec=sim_params)
p, r_stab = M_stab.theory.integrate_siegert()
# start_job(M_stab.simulation.label, submit_cmd, jobscript_template)


"""
Simulation with kappa = 1.125, chi=1.9
Presented in Fig. 4 and all following.
"""
sim_params.update({'t_sim': 100500.})
conn_params = {'g': -11.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2018/K_prime_original.npy',
               'cc_weights_factor': 1.9,
               'cc_weights_I_factor': 2.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

M_stab = MultiAreaModel(network_params, simulation=True,
                        sim_spec=sim_params)
p, r_stab = M_stab.theory.integrate_siegert()
# start_job(M_stab.simulation.label, submit_cmd, jobscript_template)


"""
Simulations with kappa = 1.125
and varying chi, presented in Fig. 4 and 8.
"""
for cc_weights_factor in [1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 2., 2.1, 2.5]:
    sim_params.update({'t_sim': 10500.})
    conn_params = {'g': -11.,
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.41666667,
                   'av_indegree_V1': 3950.,
                   'K_stable': '../SchueckerSchmidt2018/K_prime_original.npy',
                   'cc_weights_factor': cc_weights_factor,
                   'cc_weights_I_factor': 2.}

    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params}

    M_stab = MultiAreaModel(network_params, simulation=True,
                            sim_spec=sim_params)
    p, r_stab = M_stab.theory.integrate_siegert()
    # start_job(M_stab.simulation.label, submit_cmd, jobscript_template)
