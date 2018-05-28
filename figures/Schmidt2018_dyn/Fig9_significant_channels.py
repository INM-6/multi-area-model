import json
import numpy as np
import os
import sys

from collections import Counter
from helpers import structural_gradient, write_out_lw
from multiarea_model.multiarea_model import MultiAreaModel

data_path = sys.argv[1]
label = sys.argv[2]

datapath = '../../multiarea_model/data_multiarea'
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)
arch_types = proc['architecture_completed']

conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params}
M = MultiAreaModel(network_params)

gc = {}
for area in M.area_list:
    gc[area] = {}
    for pop in M.structure[area]:
        fn = os.path.join(data_path,
                          label,
                          'Analysis',
                          'granger_causality',
                          'granger_causality_{}_{}.json'.format(area, pop))
        with open(fn, 'r') as f:
            gc[area][pop] = json.load(f)


def area_pair_matrix(target_area, source_area):
    matrix = np.nan * np.zeros((len(M.structure[target_area]),
                                len(M.structure[source_area])))
    for i, target_pop in enumerate(M.structure[target_area]):
        for j, source_pop in enumerate(M.structure[source_area]):
            if source_area in gc[target_area][target_pop]:
                if source_pop in gc[target_area][target_pop][source_area]:
                    matrix[i][j] = gc[target_area][target_pop][source_area][source_pop][1]
    return np.ma.masked_where(np.isnan(matrix), matrix)


def significant_pop_pairs(target_area, source_area):
    significant_pop_pairs = []
    for i, target_pop in enumerate(gc[target_area].keys()):
        for j, source_pop in enumerate(M.structure[source_area]):
            if source_area in gc[target_area][target_pop]:
                if source_pop in gc[target_area][target_pop][source_area]:
                    if gc[target_area][target_pop][source_area][source_pop][1] < 0.05:
                        significant_pop_pairs.append((source_pop, target_pop))
    return significant_pop_pairs


"""
We detect the significant channels of causal interactions for
each pair of areas. We regard a connection a significant if the
residual variances of the vector auto-regressive models are
significant, i.e. if the p-value of the Levene test is < 0.05.

We then count the number of times a certain channel (for instance
23E-> 4E) has been detected as significant for each type of connection
and store the result in external files that will then feed into the
LaTeX scripts creating the figures of the top row of the figure.
"""

significant_channels = {'HL': [],
                        'HZ': [],
                        'LH': [],
                        'same-area': []}
for source_area in M.area_list:
    for target_area in M.area_list:
        channels = significant_pop_pairs(target_area, source_area)
        grad = structural_gradient(target_area, source_area, arch_types)
        significant_channels[grad] += channels

with open('Fig9_{}_significant_channels.json'.format(label), 'w') as f:
    json.dump(significant_channels, f)

C = Counter(significant_channels['HL'])
fn = 'Fig9_tex_files/{}_lw_HL_interactions.tex'.format(label)
write_out_lw(fn, C)

C = Counter(significant_channels['LH'])
fn = 'Fig9_tex_files/{}_lw_LH_interactions.tex'.format(label)
write_out_lw(fn, C)

C = Counter(significant_channels['HZ'])
fn = 'Fig9_tex_files/{}_lw_HZ_interactions.tex'.format(label)
write_out_lw(fn, C)

C = Counter(significant_channels['same-area'])
fn = 'Fig9_tex_files/{}_lw_local_interactions.tex'.format(label)
write_out_lw(fn, C)
