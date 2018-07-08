"""
default_parameters.py
=====================
This script defines the default values of all
parameters and defines functions to compute
single neuron and synapse parameters and to
properly set the seed of the random generators.

Authors
-------
Maximilian Schmidt
"""

from config import base_path
import json
import os

import numpy as np

complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']

population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

f1 = open(os.path.join(base_path, 'multiarea_model/data_multiarea',
                       'viscortex_raw_data.json'), 'r')
raw_data = json.load(f1)
f1.close()
av_indegree_Cragg = raw_data['av_indegree_Cragg']
av_indegree_OKusky = raw_data['av_indegree_OKusky']


"""
Simulation parameters
"""
sim_params = {
    # master seed for random number generators
    'master_seed': 0,
    # simulation step (in ms)
    'dt': 0.1,
    # simulated time (in ms)
    't_sim': 10.0,
    # no. of MPI processes:
    'num_processes': 1,
    # no. of threads per MPI process':
    'local_num_threads': 1,
    # Areas represented in the network
    'areas_simulated': complete_area_list,
}

"""
Network parameters
"""
network_params = {
    # Surface area of each area in mm^2
    'surface': 1.0,
    # Scaling of population sizes
    'N_scaling': 1.,
    # Scaling of indegrees
    'K_scaling': 1.,
    # Absolute path to the file holding full-scale rates for scaling
    # synaptic weights
    'fullscale_rates': None
}


"""
Single-neuron parameters
"""

sim_params.update(
    {
        'initial_state': {
            # mean of initial membrane potential (in mV)
            'V_m_mean': -58.0,
            # std of initial membrane potential (in mV)
            'V_m_std': 10.0
        }
    })

# dictionary defining single-cell parameters
single_neuron_dict = {
    # Leak potential of the neurons (in mV).
    'E_L': -65.0,
    # Threshold potential of the neurons (in mV).
    'V_th': -50.0,
    # Membrane potential after a spike (in mV).
    'V_reset': -65.0,
    # Membrane capacitance (in pF).
    'C_m': 250.0,
    # Membrane time constant (in ms).
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    'tau_syn_ex': 0.5,
    # Time constant of postsynaptic inhibitory currents (in ms).
    'tau_syn_in': 0.5,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0}

neuron_params = {
    # neuron model
    'neuron_model': 'iaf_psc_exp',
    # neuron parameters
    'single_neuron_dict': single_neuron_dict,
    # Mean and standard deviation for the
    # distribution of initial membrane potentials
    'V0_mean': -100.,
    'V0_sd': 50.}

network_params.update({'neuron_params': neuron_params})


"""
General connection parameters
"""
connection_params = {
    # Whether to apply the stabilization method of
    # Schuecker, Schmidt et al. (2017). Default is False.
    # Options are True to perform the stabilization or
    # a string that specifies the name of a binary
    # numpy file containing the connectivity matrix
    'K_stable': False,

    # Whether to replace all cortico-cortical connections by stationary
    # Poisson input with population-specific rates (het_poisson_stat)
    # or by time-varying current input (het_current_nonstat)
    # while still simulating all areas. In both cases, the data to replace
    # the cortico-cortical input is loaded from `replace_cc_input_source`.
    'replace_cc': False,

    # Whether to replace non-simulated areas by Poisson sources
    # with the same global rate rate_ext ('hom_poisson_stat') or
    # by specific rates ('het_poisson_stat')
    # or by time-varying specific current ('het_current_nonstat')
    # In the two latter cases, the data to replace the cortico-cortical
    # input is loaded from `replace_cc_input_source`
    'replace_non_simulated_areas': None,

    # Source of the input rates to replace cortico-cortical input
    # Either a json file (has to end on .json) holding a scalar values
    # for each population or
    # a base name such that files with names
    # $(replace_cc_input_source)-area-population.npy
    # (e.g. '$(replace_cc_input_source)-V1-23E.npy')
    # contain the time series for each population.
    # We recommend using absolute paths rather than relative paths.
    'replace_cc_input_source': None,

    # whether to redistribute CC synapse to meet literature value
    # of E-specificity
    'E_specificity': True,

    # Relative inhibitory synaptic strength (in relative units).
    'g': -16.,

    # compute average indegree in V1 from data
    'av_indegree_V1': np.mean([av_indegree_Cragg, av_indegree_OKusky]),

    # synaptic volume density
    # area-specific --> conserves average in-degree
    # constant --> conserve syn. volume density
    'rho_syn': 'constant',

    # Increase the external Poisson indegree onto 5E and 6E
    'fac_nu_ext_5E': 1.,
    'fac_nu_ext_6E': 1.,
    # to increase the ext. input to 23E and 5E in area TH
    'fac_nu_ext_TH': 1.,

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight (mV)
    'PSP_e': 0.15,
    'PSP_e_23_4': 0.3,
    # synaptic weight (mV) for external input
    'PSP_ext': 0.15,

    # relative SD of normally distributed synaptic weights
    'PSC_rel_sd_normal': 0.1,
    # relative SD of lognormally distributed synaptic weights
    'PSC_rel_sd_lognormal': 3.0,

    # scaling factor for cortico-cortical connections (chi)
    'cc_weights_factor': 1.,
    # factor to scale cortico-cortical inh. weights in relation
    # to exc. weights (chi_I)
    'cc_weights_I_factor': 1.,

    # 'switch whether to distribute weights lognormally
    'lognormal_weights': False,
    # 'switch whether to distribute only EE weight lognormally if
    # 'lognormal_weights': True
    'lognormal_EE_only': False,
}

network_params.update({'connection_params': connection_params})

"""
Delays
"""
delay_params = {
    # Local dendritic delay for excitatory transmission [ms]
    'delay_e': 1.5,
    # Local dendritic delay for inhibitory transmission [ms]
    'delay_i': 0.75,
    # Relative standard deviation for both local and inter-area delays
    'delay_rel': 0.5,
    # Axonal transmission speed to compute interareal delays [mm/ms]
    'interarea_speed': 3.5
}
network_params.update({'delay_params': delay_params})

"""
Input parameters
"""
input_params = {
    # Whether to use Poisson or DC input (True or False)
    'poisson_input': True,

    # synapse type for Poisson input
    'syn_type_ext': 'static_synapse_hpc',

    # Rate of the Poissonian spike generator (in spikes/s).
    'rate_ext': 10.,

    # Whether to switch on time-dependent DC input
    'dc_stimulus': False,
}

network_params.update({'input_params': input_params})

"""
Recording settings
"""
recording_dict = {
    # Which areas to record spike data from
    'areas_recorded': complete_area_list,

    # voltmeter
    'record_vm':  False,
    # Fraction of neurons to record membrane potentials from
    # in each population if record_vm is True
    'Nrec_vm_fraction': 0.01,

    # Parameters for the spike detectors
    'spike_dict': {
        'label': 'spikes',
        'withtime': True,
        'record_to': ['file'],
        'start': 0.},
    # Parameters for the voltmeters
    'vm_dict': {
        'label': 'vm',
        'start': 0.,
        'stop': 1000.,
        'interval': 0.1,
        'withtime': True,
        'record_to': ['file']}
    }
sim_params.update({'recording_dict': recording_dict})

"""
Theory params
"""

theory_params = {'neuron_params': neuron_params,
                 # Initial rates can be None (start integration at
                 # zero rates), a numpy.ndarray defining the initial
                 # rates or 'random_uniform' which leads to randomly
                 # drawn initial rates from a uniform distribution.
                 'initial_rates': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_iter' defines the number of
                 # different initial conditions
                 'initial_rates_iter': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_max' defines the maximum rate of the
                 # uniform distribution to draw the initial rates from
                 'initial_rates_max': 1000.,
                 # The simulation time of the mean-field theory integration
                 'T': 50.,
                 # The time step of the mean-field theory integration
                 'dt': 0.1,
                 # Time interval for recording the trajectory of the mean-field calcuation
                 # If None, then the interval is set to dt
                 'rec_interval': None}


"""
Helper function to update default parameters with custom
parameters
"""


def nested_update(d, d2):
    for key in d2:
        if isinstance(d2[key], dict) and key in d:
            nested_update(d[key], d2[key])
        else:
            d[key] = d2[key]


def check_custom_params(d, def_d):
    for key, val in d.items():
        if isinstance(val, dict):
            check_custom_params(d[key], def_d[key])
        else:
            try:
                def_val = def_d[key]
            except KeyError:
                raise KeyError('Unused key in custom parameter dictionary: {}'.format(key))
