"""
multiarea_model
==============

Network class to instantiate and administer instances of the
multi-area model of macaque visual cortex by Schmidt et al. (2018).

Classes
-------
MultiAreaModel : Loads a parameter file that specifies custom parameters for a
particular instance of the model. An instance of the model has a unique hash
label. As members, it may contain three classes:

- simulation : contains all relevant parameters for a simulation of
  the network

- theory : theory class that serves to estimate the stationary state
  of the network using mean-field theory

  Schuecker J, Schmidt M, van Albada SJ, Diesmann M, Helias M (2017)
  Fundamental Activity Constraints Lead to Specific Interpretations of
  the Connectome. PLoS Comput Biol 13(2): e1005179.
  doi:10.1371/journal.pcbi.1005179

- analysis: provides methods to load data and perform data analysis

"""
import json
import numpy as np
import os
import pprint
import shutil
from .default_params import complete_area_list, nested_update, network_params
from .default_params import check_custom_params
from collections import OrderedDict
from copy import deepcopy
from .data_multiarea.Model import compute_Model_params
from .analysis import Analysis
from config import base_path
from dicthash import dicthash
from .multiarea_helpers import (
    area_level_dict,
    load_degree_data,
    convert_syn_weight,
    dict_to_matrix,
    dict_to_vector,
    indegree_to_synapse_numbers,
    matrix_to_dict,
    vector_to_dict,
)
from .simulation import Simulation
from .theory import Theory

# Set precision of dicthash library to 1e-4
# because this is sufficient for indegrees
# and neuron numbers and guarantees reproducibility
# of the class label despite inevitably imprecise float calculations
# in the data scripts.
dicthash.FLOAT_FACTOR = 1e4
dicthash.FLOOR_SMALL_FLOATS = True


class MultiAreaModel:
    def __init__(self, network_spec, theory=False, simulation=False,
                 analysis=False, *args, **keywords):
        """
        Multiarea model class.
        An instance of the multiarea model with the given parameters.

        Parameters
        ----------
        network_spec : dict or str
            Specify the network. If it is of type dict, the parameters defined
            in the dictionary overwrite the default parameters defined in
            default_params.py.
            If it is of type str, the string defines the label of a previously
            initialized model instance that is now loaded.
        theory : bool
            whether to create an instance of the theory class as member.
        simulation : bool
            whether to create an instance of the simulation class as member.
        analysis : bool
            whether to create an instance of the analysis class as member.

        """
        self.params = deepcopy(network_params)
        if isinstance(network_spec, dict):
            print("Initializing network from dictionary.")
            check_custom_params(network_spec, self.params)
            self.custom_params = network_spec
            p_ = 'multiarea_model/data_multiarea/custom_data_files'
            # Draw random integer label for data script to avoid clashes with
            # parallelly created class instances
            rand_data_label = np.random.randint(10000)
            print("RAND_DATA_LABEL", rand_data_label)
            tmp_parameter_fn = os.path.join(base_path,
                                            p_,
                                            'custom_{}_parameter_dict.json'.format(rand_data_label))
            tmp_data_fn = os.path.join(base_path,
                                       p_,
                                       'custom_Data_Model_{}.json'.format(rand_data_label))

            with open(tmp_parameter_fn, 'w') as f:
                json.dump(self.custom_params, f)
            # Execute Data script
            compute_Model_params(out_label=str(rand_data_label),
                                 mode='custom')
        else:
            print("Initializing network from label.")
            parameter_fn = os.path.join(base_path,
                                        'config_files',
                                        '{}_config'.format(network_spec))
            tmp_data_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_Data_Model_{}.json'.format(network_spec))
            with open(parameter_fn, 'r') as f:
                self.custom_params = json.load(f)
        nested_update(self.params, self.custom_params)
        with open(tmp_data_fn, 'r') as f:
            dat = json.load(f)

        self.structure = OrderedDict()
        for area in dat['area_list']:
            self.structure[area] = dat['structure'][area]
        self.N = dat['neuron_numbers']
        self.synapses = dat['synapses']
        self.W = dat['synapse_weights_mean']
        self.W_sd = dat['synapse_weights_sd']
        self.area_list = complete_area_list
        self.distances = dat['distances']

        ind, inda, out, outa = load_degree_data(tmp_data_fn)
        # If K_stable is specified in the params, load the stabilized matrix
        # TODO: Extend this by calling the stabilization method
        if self.params['connection_params']['K_stable'] is None:
            self.K = ind
        else:
            if not isinstance(self.params['connection_params']['K_stable'], str):
                raise TypeError("Not supported. Please store the "
                                "matrix in a binary numpy file and define "
                                "the path to the file as the parameter value.")
            # Assume that the parameter defines a filename containing the matrix
            K_stable = np.load(self.params['connection_params']['K_stable'])
            ext = {area: {pop: ind[area][pop]['external'] for pop in
                          self.structure['V1']} for area in self.area_list}
            self.K = matrix_to_dict(
                K_stable, self.area_list, self.structure, external=ext)
            self.synapses = indegree_to_synapse_numbers(self.K, self.N)

        self.vectorize()
        if self.params['K_scaling'] != 1. or self.params['N_scaling'] != 1.:
            if self.params['fullscale_rates'] is None:
                raise KeyError('For downscaling, you have to define a file'
                               ' with fullscale rates.')
            self.scale_network()

        self.K_areas = area_level_dict(self.K, self.N)
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'K': self.K,
                                                       'N': self.N,
                                                       'structure': self.structure},
                                                      blacklist=[('params', 'fullscale_rates'),
                                                                 ('params',
                                                                  'connection_params',
                                                                  'K_stable'),
                                                                 ('params',
                                                                  'connection_params',
                                                                  'replace_cc_input_source')])

        if isinstance(network_spec, dict):
            parameter_fn = os.path.join(base_path,
                                        'config_files',
                                        '{}_config'.format(self.label))
            data_fn = os.path.join(base_path,
                                   'config_files',
                                   'custom_Data_Model_{}.json'.format(self.label))

            shutil.move(tmp_parameter_fn,
                        parameter_fn)
            shutil.move(tmp_data_fn,
                        data_fn)

        elif isinstance(network_spec, str):
            assert(network_spec == self.label)

        # Initialize member classes
        if theory:
            if 'theory_spec' not in keywords:
                theory_spec = {}
            else:
                theory_spec = keywords['theory_spec']
            self.init_theory(theory_spec)

        if simulation:
            if 'sim_spec' not in keywords:
                sim_spec = {}
            else:
                sim_spec = keywords['sim_spec']
            self.init_simulation(sim_spec)

        if analysis:
            assert(getattr(self, 'simulation'))
            if 'ana_spec' not in keywords:
                ana_spec = {}
            else:
                ana_spec = keywords['ana_spec']
            self.init_analysis(ana_spec)

    def __str__(self):
        s = "Multi-area network {} with custom parameters: \n".format(self.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def init_theory(self, theory_spec):
        self.theory = Theory(self, theory_spec)

    def init_simulation(self, sim_spec):
        self.simulation = Simulation(self, sim_spec)

    def init_analysis(self, ana_spec):
        assert(hasattr(self, 'simulation'))
        if 'load_areas' in ana_spec:
            load_areas = ana_spec['load_areas']
        else:
            load_areas = None
        if 'data_list' in ana_spec:
            data_list = ana_spec['data_list']
        else:
            data_list = ['spikes']
        self.analysis = Analysis(self, self.simulation,
                                 data_list=data_list,
                                 load_areas=load_areas)

    def scale_network(self):
        """
        Scale the network if `N_scaling` and/or `K_scaling` differ from 1.

        This function:
        - adjusts the synaptic weights such that the population-averaged
          stationary spike rates approximately match the given `full-scale_rates`.
        - scales the population sizes with `N_scaling` and indegrees with `K_scaling`.
        - scales the synapse numbers with `N_scaling`*`K_scaling`.
        """
        # population sizes
        self.N_vec *= self.params['N_scaling']

        # Scale the synaptic weights before the indegrees to use full-scale indegrees
        self.adj_W_to_K()
        # Then scale the indegrees and synapse numbers
        self.K_matrix *= self.params['K_scaling']
        self.syn_matrix *= self.params['K_scaling'] * self.params['N_scaling']

        # Finally recreate dictionaries
        self.N = vector_to_dict(self.N_vec, self.area_list, self.structure)
        self.K = matrix_to_dict(self.K_matrix[:, :-1], self.area_list,
                                self.structure, external=self.K_matrix[:, -1])
        self.W = matrix_to_dict(self.W_matrix[:, :-1], self.area_list,
                                self.structure, external=self.W_matrix[:, -1])

        self.synapses = matrix_to_dict(self.syn_matrix, self.area_list, self.structure)

    def vectorize(self):
        """
        Create matrix and vector version of neuron numbers, synapses
        and synapse weight dictionaries.
        """

        self.N_vec = dict_to_vector(self.N, self.area_list, self.structure)
        self.syn_matrix = dict_to_matrix(self.synapses, self.area_list, self.structure)
        self.K_matrix = dict_to_matrix(self.K, self.area_list, self.structure)
        self.W_matrix = dict_to_matrix(self.W, self.area_list, self.structure)
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])
        self.structure_vec = ['-'.join((area, pop)) for area in
                              self.area_list for pop in self.structure[area]]
        self.add_DC_drive = np.zeros_like(self.N_vec)

    def adj_W_to_K(self):
        """
        Adjust weights to scaling of neuron numbers and indegrees.

        The recurrent and external weights are adjusted to the scaling
        of the indegrees. Extra DC input is added to compensate the scaling
        and preserve the mean and variance of the input.
        """
        tau_m = self.params['neuron_params']['single_neuron_dict']['tau_m']
        C_m = self.params['neuron_params']['single_neuron_dict']['C_m']

        if isinstance(self.params['fullscale_rates'], np.ndarray):
            raise ValueError("Not supported. Please store the "
                             "rates in a file and define the path to the file as "
                             "the parameter value.")
        else:
            with open(self.params['fullscale_rates'], 'r') as f:
                d = json.load(f)
            full_mean_rates = dict_to_vector(d, self.area_list, self.structure)

        rate_ext = self.params['input_params']['rate_ext']
        J_ext = self.J_matrix[:, -1]
        K_ext = self.K_matrix[:, -1]
        x1_ext = 1e-3 * tau_m * J_ext * K_ext * rate_ext
        x1 = 1e-3 * tau_m * np.dot(self.J_matrix[:, :-1] * self.K_matrix[:, :-1], full_mean_rates)
        K_scaling = self.params['K_scaling']
        self.J_matrix /= np.sqrt(K_scaling)
        self.add_DC_drive = C_m / tau_m * ((1. - np.sqrt(K_scaling)) * (x1 + x1_ext))
        neuron_params = self.params['neuron_params']['single_neuron_dict']
        self.W_matrix = (1. / convert_syn_weight(1., neuron_params) * self.J_matrix)
