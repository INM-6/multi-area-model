"""
multiarea_helpers
==============

Helper function for the multiarea model.

Functions
---------

load_degree_data : Load indegrees and outdegrees from data file
area_level_dict : Create area-level dict from a population-level connectivity dict
dict_to_matrix : Transform dictionary of connectivity to matrix
matrix_to_dict : Transform connectivity matrix to dictionary
vector_to_dict : Transform vector of population sizes to dictionary
dict_to_vector : Transform dictionary of population sizes to vector
create_vector_mask : Create a mask for a vector of population sizes
                     to filter for specific populations.
create_mask : Create a mask for a connectivity matrix to filter for
              specific pairs of populations
indegree_to_synapse_numbers : Transform a dictionary of indegrees to a
                              a dictionary of synapse numbers

hierarchical_relation : Return the hierarchical relation of two areas
structural_gradient : Return the structural gradient of two areas
extract_area_dict :     Extract the dictionary containing only information
                        specific to a given pair of areas from a nested dictionary
                        describing the entire network.
convert_syn_weight : Convert a PSC amplitude into an integral of the PSP
"""

import json
import numpy as np
import os
from itertools import product
import collections

from config import base_path
from .default_params import complete_area_list, population_list
from nested_dict import nested_dict


def load_degree_data(fn):
    """
    Load connectivity information from json file and
    store indegrees in dictionary.

    Parameters
    ----------
    fn : string
        File name of json file. The file has to contain a dictionary
        with a subdictionary called 'synapses' containing the
        synapses between any pair of populations at the top level.

    Returns
    -------
    indegrees : dict
        Indegrees on population level. Dictionary levels are sorted as
        target area --> target population --> source area --> source population.
    indegrees_areas : dict
        Indegrees on area level. Dictionary levels are sorted as
        target area --> source area
    outdegrees : dict
        Outdegrees on population level. Dictionary levels are sorted as
        target area --> target population --> source area --> source population.
    outdegrees : dict
        Outdegrees on area level. Dictionary levels are sorted as
        target area --> source area
    """

    f = open(fn)
    dat = json.load(f)
    f.close()
    syn = dat['synapses']
    num = dat['neuron_numbers']
    indegrees = nested_dict()
    outdegrees = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(complete_area_list,
                                                                    population_list,
                                                                    complete_area_list,
                                                                    population_list):
        numT = num[target_area][target_pop]
        if numT > 0.0:
            indegrees[target_area][target_pop][source_area][source_pop] = syn[
                target_area][target_pop][source_area][source_pop] / numT
        else:
            # assign 0 to indegrees onto non-existing populations
            indegrees[target_area][target_pop][source_area][source_pop] = 0.0

        if source_area != 'external':
            numS = num[source_area][source_pop]
            if numS > 0.0:
                outdegrees[target_area][target_pop][source_area][source_pop] = syn[
                    target_area][target_pop][source_area][source_pop] / numS
            else:
                # assign 0 to outdegrees from non-existing populations
                outdegrees[target_area][target_pop][source_area][source_pop] = 0.0

    for target_area, target_pop, ext_pop in product(complete_area_list,
                                                    population_list, ['external']):
        numT = num[target_area][target_pop]
        if numT > 0.0:
            indegrees[target_area][target_pop]['external'][ext_pop] = syn[
                target_area][target_pop]['external'][ext_pop] / numT
        else:
            indegrees[target_area][target_pop]['external'][ext_pop] = 0.0

    indegrees_areas = area_level_dict(indegrees, num, degree='indegree')
    outdegrees_areas = area_level_dict(outdegrees, num, degree='outdegree')
    return (indegrees.to_dict(), indegrees_areas,
            outdegrees.to_dict(), outdegrees_areas)


def area_level_dict(dic, num, degree='indegree'):
    """
    Convert a connectivity dictionary from population-specific
    connectivity to area-level connectivity.

    Parameters
    ----------
    dic : dict
        Dictionary to transform
    num : dict
        Dictionary containing population sizes
    degree : string, {'indegree', 'outdegree'}
        Whether dic contains in- or outdegrees.
        Defaults to 'indegree'.
    """
    area_level_dic = nested_dict()
    for target_area, source_area in product(complete_area_list, complete_area_list):
        conns = 0.0
        for target_pop, source_pop in product(population_list, repeat=2):
            if degree == 'indegree':
                conns += dic[target_area][target_pop][source_area][
                    source_pop] * num[target_area][target_pop]
            elif degree == 'outdegree':
                conns += dic[target_area][target_pop][source_area][
                    source_pop] * num[source_area][source_pop]
        if degree == 'indegree':
            area_level_dic[target_area][source_area] = conns / num[target_area]['total']
        elif degree == 'outdegree':
            area_level_dic[target_area][source_area] = conns / num[source_area]['total']

    if (degree == 'indegree' and
            'external' in dic[complete_area_list[0]][population_list[0]]):
        for target_area in complete_area_list:
            conns = 0.
            for target_pop, ext_pop in product(population_list, ['external']):
                conns += dic[target_area][target_pop][
                    'external'][ext_pop] * num[target_area][target_pop]
            area_level_dic[target_area]['external'] = conns / num[target_area]['total']

    return area_level_dic.to_dict()


def dict_to_matrix(d, area_list, structure):
    """
    Convert a dictionary containing connectivity
    information of a network defined by structure to a matrix.

    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the given matrix.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dim = 0
    for area in structure.keys():
        dim += len(structure[area])

    M = np.zeros((dim, dim + 1))
    i = 0
    for target_area in area_list:
        for target_pop in structure[target_area]:
            j = 0
            for source_area in area_list:
                for source_pop in structure[source_area]:
                    M[i][j] = d[target_area][target_pop][source_area][source_pop]
                    j += 1
            M[i][j] = d[target_area][target_pop]['external']['external']
            i += 1
    return M


def matrix_to_dict(m, area_list, structure, external=None):
    """
    Convert a matrix containing connectivity
    information of a network defined by structure to a dictionary.

    Parameters
    ----------
    m : array-like
        Matrix to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the matrix to be created.
    structure : dict
        Structure of the network. Define the populations for each single area.
    external: numpy.ndarray or dict
        If None, do not include connectivity from external
        sources in the return dictionary.
        If numpy.ndarray or dict, use the connectivity given to add an entry
        'external' for each population.
        Defaults to None.
    """
    dic = nested_dict()
    for area, area2 in product(area_list, area_list):
        mask = create_mask(
            structure, target_areas=[area], source_areas=[area2], external=False)
        if external is not None:
            x = m[mask[:, :-1]]
        else:
            x = m[mask]

        if area == 'TH' and area2 == 'TH':
            x = x.reshape((6, 6))
            x = np.insert(x, 2, np.zeros((2, 6), dtype=float), axis=0)
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=1)
        elif area2 == 'TH':
            x = x.reshape((8, 6))
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=1)
        elif area == 'TH':
            x = x.reshape((6, 8))
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=0)
        else:
            x = x.reshape((8, 8))
        for i, pop in enumerate(population_list):
            for j, pop2 in enumerate(population_list):
                if x[i][j] < 1e-20:
                    x[i][j] = 0.
                dic[area][pop][area2][pop2] = x[i][j]
    if external is not None:
        if isinstance(external, np.ndarray):
            for area in dic:
                for pop in population_list:
                    if pop in structure[area]:
                        mask = create_vector_mask(
                            structure, areas=[area], pops=[pop])
                        dic[area][pop]['external'] = {
                            'external': external[mask][0]}
                    else:
                        dic[area][pop]['external'] = {
                            'external': 0.}

        if isinstance(external, dict):
            for area in dic:
                for pop in dic[area]:
                    dic[area][pop]['external'] = external[
                        area][pop]

    return dic.to_dict()


def vector_to_dict(v, area_list, structure, external=None):
    """
    Convert a vector containing neuron numbers
    of a network defined by structure to a dictionary.

    Parameters
    ----------
    v : array-like
        Vector to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the vector to be created.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dic = nested_dict()
    for area in area_list:
        vmask = create_vector_mask(structure, areas=[area])
        for i, pop in enumerate(structure[area]):
            dic[area][pop] = v[vmask][i]
        for pop in population_list:
            if pop not in structure[area]:
                dic[area][pop] = 0.

        dic[area]['total'] = sum(v[vmask])
    return dic.to_dict()


def dict_to_vector(d, area_list, structure):
    """
    Convert a dictionary containing population sizes
    of a network defined by structure to a vector.

    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the given vector.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dim = 0
    for area in structure.keys():
        dim += len(structure[area])

    V = np.zeros(dim)
    i = 0
    for target_area in area_list:
        if target_area in structure:
            for target_pop in structure[target_area]:
                if isinstance(d[target_area][target_pop], collections.Iterable):
                    V[i] = d[target_area][target_pop][0]
                else:
                    V[i] = d[target_area][target_pop]
                i += 1
    return V


def create_vector_mask(structure, pops=population_list,
                       areas=complete_area_list, complete_area_list=complete_area_list):
    """
    Create a mask for vectors to filter
    for specific populations.

    Parameters
    ----------
    structure : dict
        Structure of the network. Define the populations for each single area.
    pops : list, optinal
        List of populations for each area in the mask to be created.
        Default to population_list defined in default_params.
    areas : list, optinal
        List of areas in the mask to be created.
        Defaults to the complete_area_list defined in default_params.
    complete_area_list : list, optional
        List of areas in the network. Defines the order of areas
        in the given matrix. Defaults to the complete_area_list defined in default_params.
    """
    mask = np.array([], dtype=bool)
    for area in complete_area_list:
        if area in areas:
            mask = np.append(mask, np.in1d(np.array(structure[area]), pops))
        else:
            mask = np.append(mask, np.zeros_like(structure[area], dtype=bool))
    return mask


def create_mask(structure, target_pops=population_list,
                source_pops=population_list,
                target_areas=complete_area_list,
                source_areas=complete_area_list,
                complete_area_list=complete_area_list,
                external=True,
                **keywords):
    """
    Create a mask for the connection matrices to filter
    for specific pairs of populations.

    Parameters
    ----------
    structure : dict
        Structure of the network. Define the populations for each single area.
    target_pops : list, optinal
        List of target populations for each target area in the mask to be created.
        Default to population_list defined in default_params.
    source_pops : list, optinal
        List of source populations for each source area in the mask to be created.
        Default to population_list defined in default_params.
    target_areas : list, optinal
        List of target areas in the mask to be created.
        Defaults to the complete_area_list defined in default_params.
    source_areas : list, optinal
        List of source areas in the mask to be created.
        Defaults to the complete_area_list defined in default_params.
    complete_area_list : list, optional
        List of areas in the network. Defines the order of areas
        in the given matrix. Defaults to the complete_area_list defined in default_params.
    external : bool, optional
        Whether to include input from external source in the mask.
        Defaults to True.
    cortico_cortical : bool, optional
        Whether to filter for cortico_cortical connections only.
        Defaults to False.
    internal : bool, optional
        Whether to filter for internal connections only.
        Defaults to False.
    """
    target_mask = create_vector_mask(structure, pops=target_pops,
                                     areas=target_areas, complete_area_list=complete_area_list)
    source_mask = create_vector_mask(structure, pops=source_pops,
                                     areas=source_areas, complete_area_list=complete_area_list)
    if external:
        source_mask = np.append(source_mask, np.array([True]))
    else:
        source_mask = np.append(source_mask, np.array([False]))
    mask = np.outer(target_mask, source_mask)

    if 'cortico_cortical' in keywords and keywords['cortico_cortical']:
        negative_mask = np.zeros_like(mask, dtype=np.bool)
        for source in source_areas:
            smask = create_mask(structure,
                                target_pops=population_list,
                                target_areas=[source], source_areas=[source],
                                source_pops=population_list,
                                external=True)
            negative_mask = np.logical_or(negative_mask, smask)
        mask = np.logical_and(np.logical_not(
            np.logical_and(mask, negative_mask)), mask)
    if 'internal' in keywords and keywords['internal']:
        negative_mask = np.zeros_like(mask, dtype=np.bool)
        for source in source_areas:
            smask = create_mask(structure,
                                target_pops=population_list,
                                target_areas=[source], source_areas=[source],
                                source_pops=population_list)
            negative_mask = np.logical_or(negative_mask, smask)
        mask = np.logical_and(mask, negative_mask)
    return mask


def indegree_to_synapse_numbers(indegrees, neuron_numbers):
    """
    Transform a dictionary of indegrees to synapse numbers using
    the given neuron numbers for each population.

    Parameters
    ----------
    indegrees : dict
        Dictionary of indegrees to be transformed.
    neuron_numbers : dict
        Dictionary of population sizes
    """
    synapses = nested_dict()
    for target in indegrees:
        for tpop in indegrees[target]:
            for source in indegrees[target][tpop]:
                for spop in indegrees[target][tpop][source]:
                    synapses[target][tpop][source][spop] = indegrees[
                        target][tpop][source][spop] * neuron_numbers[target][tpop]
    return synapses.to_dict()


def hierarchical_relation(target_area, source_area):
    """
    Return the hierarchical relation between
    two areas based on their SLN value (data + estimated).
    Loads the completed SLN data from the json data file.

    Parameters
    ----------
    target_area : str
        Target area of the projection
    source_area : str
        Source area of the projection
    """
    with open(os.path.join(base_path,
                           'data_multi_area/viscortex_processed_data.json'),
              'r') as f:
        dat = json.load(f)

    SLN_completed = dat['SLN_completed']

    if target_area != source_area:
        if source_area in SLN_completed[target_area]:
            if SLN_completed[target_area][source_area] > 0.65:
                return 'FF'
            elif SLN_completed[target_area][source_area] < 0.35:
                return 'FB'
            else:
                return 'lateral'
        else:
            return None
    else:
        return 'same-area'


def structural_gradient(target_area, source_area):
    """
    Return the structural gradient between two areas.
    Loads the architectural types from the json data file.

    Parameters
    ----------
    target_area : str
        Target area of the projection
    source_area : str
        Source area of the projection
    """
    with open(os.path.join(base_path,
                           'data_multi_area/viscortex_processed_data.json'),
              'r') as f:
        dat = json.load(f)

    structure_completed = dat['structure_completed']

    if target_area != source_area:
        if structure_completed[target_area] < structure_completed[source_area]:
            return 'EL'
        elif structure_completed[target_area] > structure_completed[source_area]:
            return 'LE'
        else:
            return 'HZ'
    else:
        return 'same-area'


def extract_area_dict(d, structure, target_area, source_area):
    """
    Extract the dictionary containing only information
    specific to a given pair of areas from a nested dictionary
    describing the entire network.

    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    structure : dict
        Structure of the network. Define the populations for each single area.
    target_area : str
        Target area of the projection
    source_area : str
        Source area of the projection
    """
    area_dict = {}
    for pop in structure[target_area]:
        area_dict[pop] = {}
        for pop2 in structure[source_area]:
            area_dict[pop][pop2] = d[target_area][pop][source_area][pop2]
    return area_dict


def convert_syn_weight(W, neuron_params):
    """
    Convert the amplitude of the PSC into mV.

    Parameters
    ----------
    W : float
        Synaptic weight defined as the amplitude of the post-synaptic current.
    neuron_params : dict
        Parameters of the neuron.
    """
    tau_syn_ex = neuron_params['tau_syn_ex']
    C_m = neuron_params['C_m']
    PSP_transform = tau_syn_ex / C_m

    return PSP_transform * W
