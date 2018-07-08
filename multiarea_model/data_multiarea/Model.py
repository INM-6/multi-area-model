"""
Model
================

This script defines the model described in Schmidt et al. (2018).
The procedures are described in detail in the Methods section of
Schmidt et al. (2018).
It loads the data prepared by VisualCortexData.py and computes
neuronal numbers for each population, the external inputs
to each population and the number of synapses of each connection
in the network. These data are written out to json files.

Authors
--------
Maximilian Schmidt
Sacha van Albada

"""

import numpy as np
import json
import re
import sys
import os
import scipy
import scipy.integrate
import pprint
from copy import deepcopy
from nested_dict import nested_dict
from itertools import product
from multiarea_model.default_params import network_params, nested_update
from multiarea_model.data_multiarea.VisualCortex_Data import process_raw_data


def compute_Model_params(out_label='', mode='default'):
    """
    Compute the parameters of the network, in particular the size
    of populations, external inputs to them, and number of synapses
    in every connection.

    Parameters
    ----------
    out_label : str
        label that is appended to the output files.
    mode : str
        Mode of the function. There are three different modes:
        - default mode (mode='default')
          In default mode, all parameters are set to their default
          values defined in default_params.py .
        - custom mode (mode='custom')
          In custom mode, custom parameters are loaded from a json file
          that has to be stored in 'custom_data_files' and named as
          'custom_$(out_label)_parameter_dict.json' where $(out_label)
         is the string defined in `out_label`.
    """
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    # Load and process raw data
    process_raw_data()
    raw_fn = os.path.join(basepath, 'viscortex_raw_data.json')
    proc_fn = os.path.join(basepath, 'viscortex_processed_data.json')

    """
    Load data
    """
    with open(raw_fn, 'r') as f:
        raw_data = json.load(f)
    with open(proc_fn, 'r') as f:
        processed_data = json.load(f)

    FLN_EDR_completed = processed_data['FLN_completed']
    SLN_Data = processed_data['SLN_completed']
    Coco_Data = processed_data['cocomac_completed']
    Distance_Data = raw_data['median_distance_data']
    Area_surfaces = raw_data['surface_data']
    Intra_areal = raw_data['Intrinsic_Connectivity']
    total_thicknesses = processed_data['total_thicknesses']
    laminar_thicknesses = processed_data['laminar_thicknesses']
    Intrinsic_FLN_Data = raw_data['Intrinsic_FLN_Data']
    neuronal_numbers_fullscale = processed_data['realistic_neuronal_numbers']
    num_V1 = raw_data['num_V1']
    binzegger_data = raw_data['Binzegger_Data']

    """
    Define area and population lists.
    Define termination and origin patterns according
    to Felleman and van Essen 91
    """
    # This list of areas is ordered according
    # to their architectural type
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                 'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                 'STPa', '46', 'AITd', 'TH']
    population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    termination_layers = {'F': ['4'], 'M': ['1', '23', '5', '6'], 'C': [
        '1', '23', '4', '5', '6'], 'S': ['1', '23']}
    termination_layers2 = {'F': [4], 'M': [
        1, 2, 3, 5, 6], 'C': [1, 2, 3, 4, 5, 6], 'S': [1, 2, 3]}
    origin_patterns = {'S': ['23E'], 'I': ['5E', '6E'], 'B': ['23E', '5E', '6E']}

    binzegger_pops = list(binzegger_data.keys())
    binzegger_I_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') != -1]
    binzegger_E_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') == -1]

    # Create structure dictionary with entries for each area
    # and population that actually contains neurons
    structure = {}
    for area in area_list:
        structure[area] = []
        for pop in population_list:
            if neuronal_numbers_fullscale[area][pop] > 0.0:
                structure[area].append(pop)

    """
    If run in custom mode, load custom parameter file and
    overwrite default by custom values for parameters specified
    in the parameter file.
    """
    net_params = deepcopy(network_params)
    if mode == 'default':
        prefix = 'default'
    elif mode == 'custom':
        prefix = 'custom_data_files/custom'
        with open(os.path.join(basepath, '.'.join(('_'.join((prefix,
                                                             out_label,
                                                             'parameter_dict')),
                                                   'json'))), 'r') as f:
            custom_params = json.load(f)
        nested_update(net_params, custom_params)
        # print information on overwritten parameters
        print("\n")
        print("========================================")
        print("Customized parameters")
        print("--------------------")
        pprint.pprint(custom_params)
        print("========================================")

    """
    Define parameter values
    """
    # surface area of each area in mm^2
    surface = net_params['surface']

    conn_params = net_params['connection_params']

    # average indegree in V1 to compute
    # synaptic volume density (determined for V1 and
    # taken to be constant across areas)
    av_indegree_V1 = conn_params['av_indegree_V1']

    # Increase the external poisson indegree onto 5E and 6E
    fac_nu_ext_5E = conn_params['fac_nu_ext_5E']
    fac_nu_ext_6E = conn_params['fac_nu_ext_6E']
    # to increase the ext. input to 23E and 5E in area TH
    fac_nu_ext_TH = conn_params['fac_nu_ext_TH']

    # Single neuron parameters, important to determine synaptic weights
    single_neuron_dict = net_params['neuron_params']['single_neuron_dict']
    C_m = single_neuron_dict['C_m']
    tau_m = single_neuron_dict['tau_m']
    tau_syn_ex = single_neuron_dict['tau_syn_ex']
    tau_syn_in = single_neuron_dict['tau_syn_in']

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight (mV)
    PSP_e = conn_params['PSP_e']
    PSP_e_23_4 = conn_params['PSP_e_23_4']
    # synaptic weight (mV) for external input
    PSP_ext = conn_params['PSP_ext']
    # relative strength of inhibitory versus excitatory synapses for CUBA neurons
    g = conn_params['g']

    # relative SD of normally distributed synaptic weights
    PSC_rel_sd_normal = conn_params['PSC_rel_sd_normal']
    # relative SD of lognormally distributed synaptic weights
    PSC_rel_sd_lognormal = conn_params['PSC_rel_sd_lognormal']

    # scaling factor for cortico-cortical connections (chi)
    cc_weights_factor = conn_params['cc_weights_factor']
    # factor to scale cortico-cortical inh. weights in relation to exc. weights (chi_I)
    cc_weights_I_factor = conn_params['cc_weights_I_factor']

    # switch whether to distribute weights lognormally
    lognormal_weights = conn_params['lognormal_weights']
    # switch whether to distribute only EE weight lognormally if
    # switch_lognormal_weights = True
    lognormal_EE_only = conn_params['lognormal_EE_only']

    # whether to redistribute CC synapse to meet literature value
    # of E-specificity
    E_specificity = True

    """
    Data processing
    ===============

    Neuronal numbers
    ----------------
    """
    # Determine a synaptic volume density for each area
    rho_syn = {}
    for area in area_list:
        # note: the total thickness includes L1. Since L1 can be approximated
        # as having no neurons, rho_syn is a synapse density across all layers.
        rho_syn[area] = av_indegree_V1 * neuronal_numbers_fullscale['V1']['total'] / \
                        (Area_surfaces['V1'] * total_thicknesses['V1'])

    # Compute population sizes by scaling the realistic population
    # sizes down to the assumed area surface
    neuronal_numbers = {}
    for a in neuronal_numbers_fullscale:
        neuronal_numbers[a] = {}
        for pop in neuronal_numbers_fullscale[a]:
            neuronal_numbers[a][pop] = neuronal_numbers_fullscale[
                a][pop] / Area_surfaces[a] * surface

    """
    Intrinsic synapses
    ------------------
    The workflow is as follows:
    1. Compute the connection probabilities C'(R) of the
       microcircuit of Potjans & Diesmann (2014) depending on the area radius.
       For this, transform the connection probabilities from
       Potjans & Diesmann (2014), computed with a different
       method of averaging the Gaussian, called C_PD14 (R).
       Then, compute the in-degrees for a microcircuit with
       realistic surface and 1mm2 surface.

    2. Transform this to each area with its specific laminar
       compositions with an area-specific conversion factor
       based on the preservation of relative in-degree between
       different connections.

    3. Compute number of type I synapses.

    4. Compute number of type II synapses as the difference between
       synapses within the full-size area and the 1mm2 area.
    """

    """
    1. Radius-dependent connection probabilities of the microcircuit.
    """

    # constants for the connection probability transfer
    # from Potjans & Diesmann (2014) (PD14)
    sigma = 0.29653208289812366  # mm
    C0 = 0.1429914097112598

    # compute average connection probability with method from PD14
    r_PD14 = np.sqrt(1. / np.pi)
    C_prime_mean_PD14 = 2. / (r_PD14 ** 2) * C0 * sigma ** 2 * \
        (1. - np.exp(-r_PD14 ** 2 / (2 * sigma ** 2)))

    # New calculation based on Sheng (1985), The distance between two random
    # points in plane regions, Theorem 2.4 on the expectation value of
    # arbitrary functions of distance between points in disks.
    """
    Define integrand for Gaussian averaging
    """
    def integrand(r, R, sig):
        gauss = np.exp(-r ** 2 / (2 * sig ** 2))
        x1 = scipy.arctan(np.sqrt((2 * R - r) / (2 * R + r)))
        x2 = scipy.sin(4 * scipy.arctan(np.sqrt((2 * R - r) / (2 * R + r))))
        factor = 4 * x1 - x2
        return r * gauss * factor

    """
    Define approximation for function log(1-x) needed for large areas
    """
    def log_approx(x, limit):
        res = 0.
        for k in range(limit):
            res += x ** (k + 1) * (-1.) ** k / (k + 1)
        return res

    """
    To determine the conversion from the microcircuit model to the
    area-specific composition in our model properly, we have to
    scale down the intrinsic FLN from 0.79 to a lower value,
    detailed explanation below. Therefore, we execute the procedure
    twice: First, for realistic area size to obtain numbers for
    Indegree_prime_fullscale and then for 1mm2 areas (Indegree_prime).

    Determine mean connection probability, indegrees and intrinsic FLN
    for full-scale areas.
    """
    C_prime_fullscale_mean = {}
    for area in area_list:
        R_area = np.sqrt(Area_surfaces[area] / np.pi)
        C_prime_fullscale_mean[area] = 2 * C0 / Area_surfaces[area] * \
            scipy.integrate.quad(integrand, 0, 2 * R_area, args=(R_area, sigma))[0]

    Indegree_prime_fullscale = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list, population_list):
        C = Intra_areal[target_pop][source_pop] * \
            C_prime_fullscale_mean[area] / C_prime_mean_PD14
        if Area_surfaces[area] < 100.:  # Limit to choose between np.log and log_approx
            K = int(round(np.log(1.0 - C) / np.log(1. - 1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * Area_surfaces[area] ** 2)))) / (
                    num_V1[target_pop]['neurons'] * Area_surfaces[area])
        else:
            K = int(round(log_approx(C, 20) / log_approx(1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * Area_surfaces[area] ** 2), 20))) / (
                    num_V1[target_pop]['neurons'] * Area_surfaces[area])
        Indegree_prime_fullscale[area][target_pop][source_pop] = K
    Indegree_prime_fullscale = Indegree_prime_fullscale.to_dict()

    # Assign the average intrinsic FLN to each area
    mean_Intrinsic_FLN = Intrinsic_FLN_Data['mean']['mean']
    mean_Intrinsic_error = Intrinsic_FLN_Data['mean']['error']
    Intrinsic_FLN_completed_fullscale = {}
    for area in area_list:
        Intrinsic_FLN_completed_fullscale[area] = {
            'mean': mean_Intrinsic_FLN, 'error': mean_Intrinsic_error}

    """
    Determine mean connection probability, indegrees and intrinsic FLN
    for areas with 1mm2 surface area.
    """
    C_prime_mean = {}
    for area in area_list:
        R_area = np.sqrt(surface / np.pi)
        C_prime_mean[area] = 2 * C0 / surface * \
            scipy.integrate.quad(integrand, 0, 2 * R_area, args=(R_area, sigma))[0]

    Indegree_prime = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list, population_list):
        C = Intra_areal[target_pop][source_pop] * \
            C_prime_mean[area] / C_prime_mean_PD14
        if surface < 100.:  # Limit to choose between np.log and log_approx
            K = int(round(np.log(1.0 - C) / np.log(1. - 1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * surface ** 2)))) / (
                    num_V1[target_pop]['neurons'] * surface)
        else:
            K = int(round(log_approx(C, 20) / log_approx(1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * surface ** 2), 20))) / (
                    num_V1[target_pop]['neurons'] * surface)
        Indegree_prime[area][target_pop][source_pop] = K
    Indegree_prime = Indegree_prime.to_dict()

    Intrinsic_FLN_completed = {}
    mean_Intrinsic_FLN = Intrinsic_FLN_Data['mean']['mean']
    mean_Intrinsic_error = Intrinsic_FLN_Data['mean']['error']

    for area in area_list:
        average_relation_indegrees = []
        for pop in population_list:
            for pop2 in population_list:
                if Indegree_prime_fullscale[area][pop][pop2] > 0.:
                    average_relation_indegrees.append(Indegree_prime[
                        area][pop][pop2] / Indegree_prime_fullscale[area][pop][pop2])
        Intrinsic_FLN_completed[area] = {'mean': mean_Intrinsic_FLN * np.mean(
            average_relation_indegrees), 'error': mean_Intrinsic_error}

    """
    2. Compute the conversion factors between microcircuit
    and multi-area model areas (c_A(R)) for down-scaled and fullscale areas.
    """

    conversion_factor = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += Indegree_prime[area][target_pop][
                    source_pop] * neuronal_numbers[area][target_pop]
        conversion_factor[area] = Intrinsic_FLN_completed[area][
            'mean'] * rho_syn[area] * surface * total_thicknesses[area] / Nsyn_int_prime

    conversion_factor_fullscale = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += Indegree_prime_fullscale[area][target_pop][
                    source_pop] * neuronal_numbers_fullscale[area][target_pop]
        conversion_factor_fullscale[area] = Intrinsic_FLN_completed_fullscale[area][
            'mean'] * rho_syn[area] * Area_surfaces[area] * total_thicknesses[area] / Nsyn_int_prime

    def num_IA_synapses(area,  target_pop, source_pop, area_model='micro'):
        """
        Computes the number of intrinsic synapses from target population
        to source population in an area.

        Parameters
        ----------
        area : str
            Area for which to compute connectivity.
        target_pop : str
            Target population of the connection
        source_pop : str
            Source population of the connection
        area_model : str
            Whether to compute the number of synapses
            for the area with realistic surface area
            ('real') or 1mm2 surface area ('micro')
            Defaults to 'micro'.

        Returns
        -------
        Nsyn : float
            Number of synapses
        """
        if area_model == 'micro':
            c_area = conversion_factor[area]
            In_degree = Indegree_prime[area][
                target_pop][source_pop]
            num_source = neuronal_numbers[area][source_pop]
            num_target = neuronal_numbers[area][target_pop]
        if area_model == 'real':
            c_area = conversion_factor_fullscale[area]
            In_degree = Indegree_prime_fullscale[area][
                target_pop][source_pop]
            num_source = neuronal_numbers_fullscale[area][source_pop]
            num_target = neuronal_numbers_fullscale[area][target_pop]

        if num_source == 0 or num_target == 0:
            Nsyn = 0
        else:
            Nsyn = c_area * In_degree * num_target
        return Nsyn

    """
    3. Compute number of intrinsic (type I) synapses
    """
    synapse_numbers = nested_dict()
    for area, target_pop, source_pop in product(
            area_list, population_list, population_list):
        N_syn = num_IA_synapses(area, target_pop, source_pop)
        synapse_numbers[area][target_pop][area][source_pop] = N_syn

    # Create dictionary with total number of type I synapses for each area
    synapses_type_I = {}
    for area in area_list:
        N_syn_i = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                N_syn_i += num_IA_synapses(area, source_pop, target_pop)
        synapses_type_I[area] = N_syn_i

    """
    4. Compute number of type II synapses
    """
    synapses_type_II = {}
    s = 0.0
    for target_area in area_list:
        s_area = 0.0
        for target_pop in population_list:
            syn = 0.0
            if neuronal_numbers[target_area][target_pop] != 0.0:
                for source_pop in population_list:
                    micro_in_degree = num_IA_synapses(target_area,
                                                      target_pop, source_pop) / neuronal_numbers[
                                                          target_area][target_pop]
                    real_in_degree = (num_IA_synapses(target_area, target_pop, source_pop,
                                                      area_model='real')
                                      / neuronal_numbers_fullscale[
                                                         target_area][target_pop])
                    syn += (real_in_degree - micro_in_degree) * \
                        neuronal_numbers[target_area][target_pop]
            s_area += syn
        synapses_type_II[target_area] = s_area

    """
    Cortico-cortical synapses
    ------------------
    1. Normalize FLN values of cortico-cortical connection
       to (1 - FLN_i - 0.013).
       1.3%: subcortical inputs, data from Markov et al. (2011)
    """
    FLN_completed = {}
    for target_area in FLN_EDR_completed:
        FLN_completed[target_area] = {}
        cc_proportion = (1.-Intrinsic_FLN_completed_fullscale[target_area]['mean']-0.013)
        norm_factor = cc_proportion / sum(FLN_EDR_completed[target_area].values())
        for source_area in FLN_EDR_completed[target_area]:
            FLN_completed[target_area][source_area] = norm_factor * FLN_EDR_completed[
                target_area][source_area]

    """
    2. Process Binzegger data
       The notation follows Eqs. (11-12 and following) in
       Schmidt et al. (2018):
       v : layer of cortico-cortical synapse
       cb : cell type
       cell_layer : layer of the cell
       i : population in the model
    """

    # Determine the relative numbers of the 8 populations in Binzegger's data
    relative_numbers_binzegger = {'23E': 0.0, '23I': 0.0,
                                  '4E': 0.0, '4I': 0.0,
                                  '5E': 0.0, '5I': 0.0,
                                  '6E': 0.0, '6I': 0.0}
    s = 0.0
    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if cell_layer not in ['', '1']:
            s += binzegger_data[cb]['occurrence']

    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if cell_layer not in ['', '1']:
            if cb in binzegger_E_pops:
                relative_numbers_binzegger[
                    cell_layer + 'E'] += binzegger_data[cb]['occurrence'] / s
            if cb in binzegger_I_pops:
                relative_numbers_binzegger[
                    cell_layer + 'I'] += binzegger_data[cb]['occurrence'] / s

    # Determine the relative numbers of the 8 populations in V1
    relative_numbers_model = {'23E': 0.0, '23I': 0.0,
                              '4E': 0.0, '4I': 0.0,
                              '5E': 0.0, '5I': 0.0,
                              '6E': 0.0, '6I': 0.0}

    for pop in neuronal_numbers['V1']:
        relative_numbers_model[pop] = neuronal_numbers[
            'V1'][pop] / neuronal_numbers['V1']['total']

    # Process Binzegger data into conditional probabilities: What is the
    # probability of having a cell body in layer u if a cortico-cortical
    # connection forms a synapse in layer v ?

    # Compute number of CC synapses formed in each layer
    num_cc_synapses = {'1': 0.0, '23': 0.0, '4': 0.0, '5': 0.0, '6': 0.0}
    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if cb in binzegger_E_pops:
            i = cell_layer + 'E'
        if cb in binzegger_I_pops:
            i = cell_layer + 'I'
        if i != '1I':
            for v in binzegger_data[cb]['syn_dict']:
                if v in num_cc_synapses:
                    num_ratio = relative_numbers_model[i] / relative_numbers_binzegger[i]
                    cc_syn_num = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
                                  binzegger_data[cb]['syn_dict'][v][
                                      'number of synapses per neuron'] *
                                  binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)

                    num_cc_synapses[v] += cc_syn_num

    # Compute cond. probability
    synapse_to_cell_body_basis = {}
    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if cb in binzegger_E_pops:
            i = cell_layer + 'E'
        else:
            i = cell_layer + 'I'
        for v in binzegger_data[cb]['syn_dict']:
            if v in num_cc_synapses:
                if i != '1I':  # We do not model cell types in layer 1
                    num_ratio = relative_numbers_model[i] / relative_numbers_binzegger[i]
                    value = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
                             binzegger_data[cb]['syn_dict'][v]['number of synapses per neuron'] *
                             binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)
                    cond_prob = value / num_cc_synapses[v]
                    if v in synapse_to_cell_body_basis:
                        if i in synapse_to_cell_body_basis[v]:
                            synapse_to_cell_body_basis[
                                v][i] += cond_prob
                        else:
                            synapse_to_cell_body_basis[
                                v].update({i: cond_prob})
                    else:
                        synapse_to_cell_body_basis.update(
                            {v: {i: cond_prob}})

    # Make synapse_to_cell_body area-specific to account for
    # missing layers in some areas (area TH)
    synapse_to_cell_body = {}
    for area in area_list:
        synapse_to_cell_body[area] = deepcopy(synapse_to_cell_body_basis)

    for layer in synapse_to_cell_body['TH']:
        l = 0.
        for pop in ['23E', '5E', '6E']:
            l += laminar_thicknesses['TH'][pop[0:-1]]
        for pop in ['23E', '5E', '6E']:
            if '4E' in synapse_to_cell_body['TH'][layer]:
                if pop in synapse_to_cell_body['TH'][layer]:
                    synapse_to_cell_body['TH'][layer][pop] += synapse_to_cell_body[
                        'TH'][layer]['4E'] * laminar_thicknesses['TH'][pop[0:-1]] / l
                else:
                    synapse_to_cell_body['TH'][layer][pop] = synapse_to_cell_body[
                        'TH'][layer]['4E'] * laminar_thicknesses['TH'][pop[0:-1]] / l
        l = 0.
        for pop in ['23I', '5I', '6I']:
            l += laminar_thicknesses['TH'][pop[0:-1]]
        for pop in ['23I', '5I', '6I']:
            if '4I' in synapse_to_cell_body['TH'][layer]:
                if pop in synapse_to_cell_body['TH'][layer]:
                    synapse_to_cell_body['TH'][layer][pop] += synapse_to_cell_body[
                        'TH'][layer]['4I'] * laminar_thicknesses['TH'][pop[0:-1]] / l
                else:
                    synapse_to_cell_body['TH'][layer][pop] = synapse_to_cell_body[
                        'TH'][layer]['4I'] * laminar_thicknesses['TH'][pop[0:-1]] / l

    for layer in synapse_to_cell_body['TH']:
        if '4E' in synapse_to_cell_body['TH'][layer]:
            del synapse_to_cell_body['TH'][layer]['4E']
        if '4I' in synapse_to_cell_body['TH'][layer]:
            del synapse_to_cell_body['TH'][layer]['4I']

    def num_CC_synapses(target_area, target_pop, source_area, source_pop):
        """
        Compute number of synapses between two populations in different areas

        Parameters
        ----------
        target_area : str
            Target area of the connection
        target_pop : str
            Target population of the connection
        source_area : str
            Source area of the connection
        source_pop : str
            Source population of the connection

        Returns
        -------
        Nsyn : float
            Number of synapses of the connection.
        """

        Nsyn = 0.0

        # Test if the connection exists.
        if (source_area in Coco_Data[target_area] and
            source_pop not in ['4I', '4E'] and
            neuronal_numbers[target_area][target_pop] != 0 and
                source_pop not in ['23I', '4I', '5I', '6I']):

            num_source = neuronal_numbers_fullscale[source_area][source_pop]

            # information on the area level
            FLN_BA = FLN_completed[target_area][source_area]
            Nsyn_tot = rho_syn[target_area] * \
                Area_surfaces[target_area] * total_thicknesses[target_area]

            # source side
            # if there is laminar information in CoCoMac, use it
            if Coco_Data[target_area][source_area]['source_pattern'] is not None:
                sp = np.array(Coco_Data[target_area][source_area][
                              'source_pattern'], dtype=np.float)

                # Manually determine SLN, based on CoCoMac:
                # from supragranular, then SLN=0.,
                # no connections from infragranular --> SLN=1.
                if np.all(sp[:3] == 0):
                    SLN_value = 0.
                elif np.all(sp[-2:] == 0):
                    SLN_value = 1.
                else:
                    SLN_value = SLN_Data[target_area][source_area]

                if source_pop in origin_patterns['S']:
                    if np.any(sp[:3] != 0):
                        X = SLN_value
                        Y = 1.  # Only layer 2/3 is part of the supragranular pattern
                    else:
                        X = 0.
                        Y = 0.

                elif source_pop in origin_patterns['I']:
                    if np.any(sp[-2:] != 0):
                        # Distribute between 5 and 6 according to CocoMac values
                        index = list(range(1, 7)).index(int(source_pop[:-1]))
                        if sp[index] != 0:
                            X = 1. - SLN_value
                            Y = 10 ** (sp[index]) / np.sum(10 **
                                                           sp[-2:][np.where(sp[-2:] != 0)])
                        else:
                            X = 0.
                            Y = 0.
                    else:
                        X = 0.
                        Y = 0.
            # otherwise, use neuronal numbers
            else:
                if source_pop in origin_patterns['S']:
                    X = SLN_Data[target_area][source_area]
                    Y = 1.0  # Only layer 2/3 is part of the supragranular pattern

                elif source_pop in origin_patterns['I']:
                    X = 1.0 - SLN_Data[target_area][source_area]
                    infra_neurons = 0.0
                    for i in origin_patterns['I']:
                        infra_neurons += neuronal_numbers_fullscale[
                            source_area][i]
                    Y = num_source / infra_neurons

            # target side
            # if there is laminar data in CoCoMac, use this
            if Coco_Data[target_area][source_area]['target_pattern'] is not None:
                tp = np.array(Coco_Data[target_area][source_area][
                              'target_pattern'], dtype=np.float)

                # If there is a '?' (=-1) in the data, check if this layer is in
                # the termination pattern induced by hierarchy and insert a 2 if
                # yes
                if -1 in tp:
                    if (SLN_Data[target_area][source_area] > 0.35 and
                            SLN_Data[target_area][source_area] <= 0.65):
                        T_hierarchy = termination_layers2['C']
                    elif SLN_Data[target_area][source_area] < 0.35:
                        T_hierarchy = termination_layers2['M']
                    elif SLN_Data[target_area][source_area] > 0.65:
                        T_hierarchy = termination_layers2['F']
                    for l in T_hierarchy:
                        if tp[l - 1] == -1:
                            tp[l - 1] = 2
                T = np.where(tp > 0.)[0] + 1  # '+1' transforms indices to layers
                # Here we treat the values as numbers of labeled neurons rather
                # than densities for the sake of simplicity
                p_T = np.sum(10 ** tp[np.where(tp > 0.)[0]])
                Nsyn = 0.0
                su = 0.
                for i in range(len(T)):
                    if T[i] in [2, 3]:
                        syn_layer = '23'
                    else:
                        syn_layer = str(T[i])
                    Z = 10 ** tp[np.where(tp > 0.)[0]][i] / p_T
                    if target_pop in synapse_to_cell_body[target_area][syn_layer]:
                        Nsyn += synapse_to_cell_body[target_area][syn_layer][
                            target_pop] * Nsyn_tot * FLN_BA * X * Y * Z

                    su += Z

            # otherwise use laminar thicknesses
            else:
                if (SLN_Data[target_area][source_area] > 0.35 and
                        SLN_Data[target_area][source_area] <= 0.65):
                    T = termination_layers['C']
                elif SLN_Data[target_area][source_area] < 0.35:
                    T = termination_layers['M']
                elif SLN_Data[target_area][source_area] > 0.65:
                    T = termination_layers['F']

                p_T = 0.0
                for i in T:
                    if i != '1':
                        p_T += laminar_thicknesses[target_area][i]

                Nsyn = 0.0
                for syn_layer in T:
                    if target_pop in synapse_to_cell_body[target_area][syn_layer]:
                        if syn_layer == '1':
                            Z = 0.5
                        else:
                            if '1' in T:
                                Z = 0.5 * \
                                    laminar_thicknesses[
                                        target_area][syn_layer] / p_T
                            else:
                                Z = laminar_thicknesses[
                                    target_area][syn_layer] / p_T
                        Nsyn += synapse_to_cell_body[target_area][syn_layer][
                            target_pop] * Nsyn_tot * FLN_BA * X * Y * Z

        return Nsyn

    """
    Compute the number of cortico-cortical synapses
    for each pair of populations.
    """
    # area TH does not have a granular layer
    neuronal_numbers_fullscale['TH']['4E'] = 0.0
    neuronal_numbers['TH']['4E'] = 0.0
    neuronal_numbers_fullscale['TH']['4I'] = 0.0
    neuronal_numbers['TH']['4I'] = 0.0

    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,
                                                                    area_list, population_list):
        if target_area != source_area:
            N_fullscale = neuronal_numbers_fullscale[target_area][target_pop]
            N = neuronal_numbers[target_area][target_pop]
            if N != 0:
                N_syn = num_CC_synapses(target_area, target_pop,
                                        source_area, source_pop) / N_fullscale * N
            else:
                N_syn = 0.0
            synapse_numbers[target_area][target_pop][source_area][source_pop] = N_syn

    synapse_numbers = synapse_numbers.to_dict()

    """
    If switch_E_specificity is True, redistribute
    the synapses of feedback connections to achieve
    the E_specific_factor of 0.93
    """
    if E_specificity:
        E_specific_factor = 0.93
        for target_area in area_list:
            for source_area in area_list:
                if (target_area != source_area and source_area in Coco_Data[target_area] and
                        SLN_Data[target_area][source_area] < 0.35):
                    syn_I = 0.0
                    syn_E = 0.0
                    for target_pop in synapse_numbers[target_area]:
                        for source_pop in synapse_numbers[target_area][target_pop][source_area]:
                            if target_pop.find('E') > -1:
                                syn_E += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                            else:
                                syn_I += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                    if syn_E > 0.0 or syn_I > 0.0:
                        alpha_E = syn_E / (syn_E + syn_I)
                        alpha_I = syn_I / (syn_E + syn_I)
                        if alpha_I != 0.0 and alpha_E != 0.0:
                            for target_pop in synapse_numbers[target_area]:
                                for source_pop in synapse_numbers[target_area][
                                        target_pop][source_area]:
                                    N_syn = synapse_numbers[target_area][target_pop][
                                        source_area][source_pop]
                                    if target_pop.find('E') > -1:
                                        synapse_numbers[target_area][target_pop][source_area][
                                            source_pop] = E_specific_factor / alpha_E * N_syn
                                    else:
                                        synapse_numbers[target_area][target_pop][source_area][
                                            source_pop] = (1. - E_specific_factor) / alpha_I * N_syn

    """
    External inputs
    ---------------
    To determine the number of external inputs to each
    population, we compute the total number of external
    to an area and then distribute the synapses such that
    each population receives the same indegree from external
    Poisson sources.


    1. Compute the total number of external synapses to each
       area as the difference between the total number of
       synapses and the intrinsic (type I) and cortico-cortical
       (type III) synapses.
    """
    External_synapses = {}
    for target_area in area_list:
        N_syn_tot = surface * total_thicknesses[target_area] * rho_syn[target_area]
        CC_synapses = 0.0
        for target_pop, source_area, source_pop in product(population_list, area_list,
                                                           population_list):
            if source_area != target_area:
                CC_synapses += synapse_numbers[target_area][
                    target_pop][source_area][source_pop]
        ext_syn = N_syn_tot * (1. - Intrinsic_FLN_completed[target_area]['mean']) - CC_synapses
        External_synapses[target_area] = ext_syn

    """
    2. Distribute poisson sources among populations such that each
       population receives the same Poisson indegree.
       For this, we construct a system of linear equations and solve
       this using a least-squares algorithm (numpy.linalg.lstsq).
    """
    for area in area_list:
        nonvisual_fraction_matrix = np.zeros(
            (len(structure[area]) + 1, len(structure[area])))
        for i in range(len(structure[area])):
            nonvisual_fraction_matrix[
                i] = 1. / len(structure[area]) * np.ones(len(structure[area]))
            nonvisual_fraction_matrix[i][i] -= 1

        for i in range(len(structure[area])):
            nonvisual_fraction_matrix[-1][
                i] = neuronal_numbers[area][structure[area][i]]

        vector = np.zeros(len(structure[area]) + 1)
        ext_syn = External_synapses[area]
        vector[-1] = ext_syn
        solution, residues, rank, s = np.linalg.lstsq(
            nonvisual_fraction_matrix, vector)
        for i, pop in enumerate(structure[area]):
            synapse_numbers[area][pop]['external'] = {
                'external': solution[i] * neuronal_numbers[area][pop]}

    synapse_numbers['TH']['4E']['external'] = {'external': 0.0}
    synapse_numbers['TH']['4I']['external'] = {'external': 0.0}

    """
    Modify external inputs according to additional factors
    """
    for target_area in area_list:
        for target_pop in synapse_numbers[target_area]:
            if target_pop in ['5E']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_5E * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['6E']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_6E * synapse_numbers[target_area][target_pop][
                        'external']['external']

    synapse_numbers['TH']['23E']['external']['external'] *= fac_nu_ext_TH
    synapse_numbers['TH']['5E']['external']['external'] *= fac_nu_ext_TH

    """
    Synaptic weights
    ----------------
    Create dictionaries with the mean and standard deviation
    of the synaptic weight of each connection in the network.
    Depends on the chosen neuron model.
    """

    # for current-based neurons
    PSC_e_over_PSP_e = ((C_m**(-1) * tau_m * tau_syn_ex / (tau_syn_ex - tau_m) *
                         ((tau_m / tau_syn_ex) ** (- tau_m / (tau_m - tau_syn_ex)) -
                          (tau_m / tau_syn_ex) ** (- tau_syn_ex / (tau_m - tau_syn_ex)))) ** (-1))
    PSC_i_over_PSP_i = ((C_m ** (-1) * tau_m * tau_syn_in / (tau_syn_in - tau_m) *
                         ((tau_m / tau_syn_in) ** (- tau_m / (tau_m - tau_syn_in)) -
                          (tau_m / tau_syn_in) ** (- tau_syn_in / (tau_m - tau_syn_in)))) ** (-1))

    synapse_weights_mean = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,
                                                                    area_list, population_list):
        if 'E' in source_pop:
            synapse_weights_mean[target_area][target_pop][source_area][
                source_pop] = PSC_e_over_PSP_e * PSP_e
        else:
            synapse_weights_mean[target_area][target_pop][source_area][
                source_pop] = PSC_i_over_PSP_i * g * PSP_e

    synapse_weights_mean = synapse_weights_mean.to_dict()
    synapse_weights_sd = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,
                                                                    area_list, population_list):
        mean = abs(synapse_weights_mean[target_area][target_pop][source_area][source_pop])
        if ((lognormal_weights and 'E' in target_pop and 'E' in source_pop) or
                lognormal_weights and not lognormal_EE_only):
            sd = PSC_rel_sd_lognormal * mean
        else:
            sd = PSC_rel_sd_normal * mean
        synapse_weights_sd[target_area][target_pop][source_area][source_pop] = sd
    synapse_weights_sd = synapse_weights_sd.to_dict()

    # Apply specific weight for intra_areal 4E-->23E connections
    for area in area_list:
        synapse_weights_mean[area]['23E'][area]['4E'] = PSP_e_23_4 * PSC_e_over_PSP_e
        synapse_weights_sd[area]['23E'][area]['4E'] = (PSC_rel_sd_normal * PSP_e_23_4
                                                       * PSC_e_over_PSP_e)

    # Apply cc_weights_factor for all CC connections
    for target_area, source_area in product(area_list, area_list):
        if source_area != target_area:
            for target_pop, source_pop in product(population_list, population_list):
                synapse_weights_mean[target_area][target_pop][
                    source_area][source_pop] *= cc_weights_factor
                synapse_weights_sd[target_area][target_pop][
                    source_area][source_pop] *= cc_weights_factor

    # Apply cc_weights_I_factor for all CC connections
    for target_area, source_area in product(area_list, area_list):
        if source_area != target_area:
            for target_pop, source_pop in product(population_list, population_list):
                if 'I' in target_pop:
                    synapse_weights_mean[target_area][target_pop][
                        source_area][source_pop] *= cc_weights_I_factor
                    synapse_weights_sd[target_area][target_pop][
                        source_area][source_pop] *= cc_weights_I_factor

    # Synaptic weights for external input
    for target_area in area_list:
        for target_pop in population_list:
            synapse_weights_mean[target_area][target_pop]['external'] = {
                'external': PSC_e_over_PSP_e * PSP_ext}

    """
    Output section
    --------------
    All data are saved to a json file with the name structure:
    '$(prefix) + '_Data_Model' + $(out_label) + .json'.
    """

    collected_data = {'area_list': area_list,
                      'av_indegree_V1': av_indegree_V1,
                      'population_list': population_list,
                      'structure': structure,
                      'synapses_orig': synapse_numbers,
                      'synapses': synapse_numbers,
                      'realistic_neuron_numbers': neuronal_numbers_fullscale,
                      'realistic_synapses': synapse_numbers,
                      'neuron_numbers': neuronal_numbers,
                      'synapses_type_I': synapses_type_I,
                      'synapses_type_II': synapses_type_II,
                      'distances': Distance_Data,
                      'binzegger_processed': synapse_to_cell_body,
                      'Intrinsic_FLN_completed': Intrinsic_FLN_completed,
                      'synapse_weights_mean': synapse_weights_mean,
                      'synapse_weights_sd': synapse_weights_sd
                      }

    with open(os.path.join(basepath,
                           '.'.join(('_'.join((prefix,
                                               'Data_Model',
                                               out_label)),
                                     'json'))), 'w') as f:
        json.dump(collected_data, f)


if __name__ == '__main__':
    compute_Model_params()
