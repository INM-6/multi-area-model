import json
import networkx as nx
import numpy as np
import os

from collections import Counter
from itertools import product
from helpers import original_data_path
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask, dict_to_vector

from helpers import structural_gradient, write_out_lw
from helpers import area_population_list

import sys
sys.path.append('../Schmidt2018/')
from graph_helpers import all_pairs_bellman_ford_path
from graph_helpers import create_networkx_area_graph, create_networkx_graph

data_path = sys.argv[1]
label = sys.argv[2]

"""
Load data and create MultiAreaModel instance
"""
datapath = '../../multiarea_model/data_multiarea'
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)
arch_types = proc['architecture_completed']

initial_rates = np.zeros(254)
par = {'connection_params': {'g': -11.,
                             'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy',
                             'cc_weights_I_factor': 2.,
                             'cc_weights_factor': 1.9,
                             'fac_nu_ext_5E': 1.125,
                             'fac_nu_ext_6E': 1.125 * 10 / 3. - 7 / 3.,
                             'fac_nu_ext_TH': 1.2},
       'input_params': {'rate_ext': 10.}}
theory_params = {'T': 50.,
                 'dt': 0.1,
                 'initial_rates': initial_rates}

M = MultiAreaModel(par, theory=True, simulation=False, theory_spec=theory_params)
# pops, rates_full = M.theory.integrate_siegert()

# stationary firing rates: We use stationary rates computed during a
# phase where the network state shows low activity (between 3500. and
# 3600. ms)
fn = os.path.join(data_path, label, 'Analysis', 'pop_rates_LA_state.json')
with open(fn, 'r') as f:
    pop_rates = json.load(f)
rates = dict_to_vector(pop_rates, M.area_list, M.structure)

# Construct gain matrix with absolute values of synaptic weights
mu, sigma = M.theory.mu_sigma(rates)
d_nu_d_mu, d_nu_d_sigma = M.theory.d_nu(mu, sigma)
d_nu_d_mu_matrix = np.zeros_like(M.K_matrix[:, :-1])
d_nu_d_sigma_matrix = np.zeros_like(M.K_matrix[:, :-1])
for i in range(len(d_nu_d_mu)):
    d_nu_d_mu_matrix[:, i] = d_nu_d_mu
    d_nu_d_sigma_matrix[:, i] = d_nu_d_sigma

gain_matrix = (M.K_matrix[:, :-1] * np.absolute(M.J_matrix[:, :-1]) * d_nu_d_mu_matrix +
               M.K_matrix[:, :-1] * M.J_matrix[:, :-1]**2 * d_nu_d_sigma_matrix)
eig = np.linalg.eig(gain_matrix)
gain_matrix_rescaled = gain_matrix / np.max(np.real(eig[0]))

# Create population-level graph and determine paths and path lengths
g = create_networkx_graph(gain_matrix_rescaled, M.structure_vec, relative=False)
paths, path_lengths = all_pairs_bellman_ford_path(g, weight='distance')

# Treat area MDP which does not receive connections from other areas
for area in M.area_list:
    for target_pop in area_population_list(M.structure, area):
        for source_pop in area_population_list(M.structure, 'MDP'):
            path_lengths[target_pop][source_pop] = np.inf
            paths[target_pop][source_pop] = []

path_length_matrix = np.zeros((254, 254))
for i, source in enumerate(M.structure_vec):
    for j, target in enumerate(M.structure_vec):
        if target in path_lengths[source]:
            path_length_matrix[j][i] = path_lengths[source][target]
        else:
            path_length_matrix[j][i] = np.inf

# Create dictionary containing the shortest path between any pair of areas
CC_paths = {area: {} for area in M.area_list}
for target_area in M.area_list:
    for source_area in M.area_list:
        if target_area != source_area:
            pop_iter = product(area_population_list(M.structure, target_area),
                               area_population_list(M.structure, source_area))
            path_list = []
            path_length_list = []
            for tpop, spop in pop_iter:
                path_list.append(paths[spop][tpop])
                path_length_list.append(path_lengths[spop][tpop])
            CC_paths[source_area][target_area] = path_list[np.argmin(
                path_length_list)]


"""
Analyze the paths between areas for different types of connections
and write out the linewidths for the corresponding tex plots to file.
"""

HL_path_pairs = []
LH_path_pairs = []
HZ_path_pairs = []

HL_path_lengths = []
LH_path_lengths = []
HZ_path_lengths = []

HL_paths = []
LH_paths = []
HZ_paths = []

for target_area in M.area_list:
    for source_area in M.area_list:
        indices = create_mask(M.structure, target_pops=M.structure[target_area],
                              source_pops=M.structure[source_area],
                              target_areas=[target_area],
                              source_areas=[source_area],
                              complete_area_list=M.area_list,
                              external=False)
        pM = path_length_matrix[indices[:, :-1]]
        pM = pM.reshape(
            (len(M.structure[target_area]), len(M.structure[source_area])))
        imin = np.unravel_index(np.argmin(pM), pM.shape)
        pair = (M.structure[source_area][imin[1]],
                M.structure[target_area][imin[0]])

        source_pop = area_population_list(
            M.structure, source_area)[imin[1]]
        target_pop = area_population_list(
            M.structure, target_area)[imin[0]]
        if target_area != source_area and len(paths[source_pop][target_pop]) > 0:
            if structural_gradient(target_area, source_area, arch_types) == 'LH':
                LH_path_pairs.append(pair)
                LH_paths.append(paths[source_pop][target_pop])
                LH_path_lengths.append(
                    path_lengths[source_pop][target_pop])
            elif structural_gradient(target_area, source_area, arch_types) == 'HL':
                HL_path_pairs.append(pair)
                HL_paths.append(paths[source_pop][target_pop])
                HL_path_lengths.append(
                    path_lengths[source_pop][target_pop])
            else:
                HZ_path_pairs.append(pair)
                HZ_paths.append(paths[source_pop][target_pop])
                HZ_path_lengths.append(
                    path_lengths[source_pop][target_pop])

C = Counter(HL_path_pairs)
fn = 'Fig9_tex_files/{}_lw_HL_paths.tex'.format(label)
write_out_lw(fn, C)

C = Counter(LH_path_pairs)
fn = 'Fig9_tex_files/{}_lw_LH_paths.tex'.format(label)
write_out_lw(fn, C)

C = Counter(HZ_path_pairs)
fn = 'Fig9_tex_files/{}_lw_HZ_paths.tex'.format(label)
write_out_lw(fn, C)

C = Counter(HL_path_pairs)
