import copy

from multiarea_model import MultiAreaModel
from start_jobs import start_job
from config import submit_cmd, jobscript_template

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

params_LA = (network_params, copy.deepcopy(sim_params))

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

params_HA = (network_params, copy.deepcopy(sim_params))


params_stab = {}
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
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}
params_stab[1.0] = (network_params, copy.deepcopy(sim_params))

"""
Simulation with kappa = 1.125, chi=1.9
Presented in Fig. 4 and all following.

One simulation with t_sim=10500. ms to plot spike raster plots (to
reduce the computational load when loading spikes) and one with
t_sim=100500. ms for all quantitative measures.
"""
sim_params.update({'t_sim': 10500.})
conn_params = {'g': -11.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy',
               'cc_weights_factor': 1.9,
               'cc_weights_I_factor': 2.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}
params_stab['1.9_spikes'] = (network_params, copy.deepcopy(sim_params))

sim_params.update({'t_sim': 100500.})
conn_params = {'g': -11.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy',
               'cc_weights_factor': 1.9,
               'cc_weights_I_factor': 2.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}
params_stab[1.9] = (network_params, copy.deepcopy(sim_params))


"""
Simulations with kappa = 1.125
and varying chi, presented in Fig. 4 and 8.
"""
cc_weights_factor_list = [1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 2., 2.1, 2.5]
for cc_weights_factor in cc_weights_factor_list:
    sim_params.update({'t_sim': 10500.})
    conn_params = {'g': -11.,
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.41666667,
                   'av_indegree_V1': 3950.,
                   'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy',
                   'cc_weights_factor': cc_weights_factor,
                   'cc_weights_I_factor': 2.}

    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params}

    params_stab[cc_weights_factor] = (network_params, copy.deepcopy(sim_params))

"""
Collect all parameter dictionaries in one place
"""
NEW_SIM_PARAMS = {'all': [params_LA,
                          params_HA]
                  + [params_stab[chi] for chi in params_stab],
                  'Fig1': None,
                  'Fig2': [params_LA,
                           params_HA,
                           params_stab[1.0]],
                  'Fig3': [params_stab[1.0]],
                  'Fig4': [params_stab[chi] for chi in [1., 1.8, 1.9, 2., 2.1, 2.5]],
                  'Fig5': [params_stab['1.9_spikes'],
                           params_stab[1.9]],
                  'Fig6': [params_stab[chi] for chi in [1., 2.5, '1.9_spikes', 1.9]],
                  'Fig7': [params_stab[1.9]],
                  'Fig8': [params_stab[chi] for chi in [1., 1.4, 1.5, 1.6, 1.7, 1.75,
                                                        1.8, 2., 2.1, 2.5, 1.9]],
                  'Fig9': [params_stab[1.9]]}

"""
Collect all labels in one dictionary
"""


def init_model(par):
    return MultiAreaModel(par[0],
                          simulation=True,
                          sim_spec=par[1])


def init_models(figure):
    """
    Create instances of the models required for the given figure.
    """
    models = []
    for par in NEW_SIM_PARAMS[figure]:
        M = init_model(par)
        models.append(M)
    return models


def create_label_dict():
    M_LA = init_model(NEW_SIM_PARAMS['all'][0])
    M_HA = init_model(NEW_SIM_PARAMS['all'][1])
    M_stab = {chi: init_model(params_stab[chi]) for chi in params_stab}
    NEW_SIM_LABELS = {'all': [M_LA.simulation.label,
                              M_HA.simulation.label]
                      + [M_stab[chi].simulation.label for chi in M_stab.keys()],
                      'Fig1': None,
                      'Fig2': [M_LA.simulation.label,
                               M_HA.simulation.label,
                               M_stab[1.0].simulation.label],
                      'Fig3': [M_stab[1.0].simulation.label],
                      'Fig4': [M_stab[chi].simulation.label for chi in [1., 1.8, 1.9,
                                                                        2., 2.1, 2.5]],
                      'Fig5': [M_stab['1.9_spikes'].simulation.label,
                               M_stab[1.9].simulation.label],
                      'Fig6': [M_stab[chi].simulation.label for chi in [1., 2.5, 1.9,
                                                                        '1.9_spikes']],
                      'Fig7': [M_stab[1.9].simulation.label],
                      'Fig8': [M_stab[chi].simulation.label for chi in [1., 1.4, 1.5, 1.6,
                                                                        1.7, 1.75, 1.8, 2.,
                                                                        2.1, 2.5, 1.9]],
                      'Fig9': [M_stab[1.9].simulation.label]}
    return NEW_SIM_LABELS


def run_simulation(figure):
    models = init_models(figure)
    for M in models:
        start_job(M.simulation.label, submit_cmd, jobscript_template)
    
    
