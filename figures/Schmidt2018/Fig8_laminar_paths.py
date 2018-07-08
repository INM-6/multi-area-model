import json
import matplotlib.pyplot as pl
import networkx as nx
import numpy as np
import os

from collections import Counter
from graph_helpers import all_pairs_bellman_ford_path
from graph_helpers import create_networkx_area_graph, create_networkx_graph
from helpers import area_list, area_population_list
from helpers import datapath
from itertools import product
from helpers import structural_gradient, hierarchical_relation, write_out_lw
from matplotlib import rc_file
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask
from plotcolors import myblue

rc_file('plotstyle.rc')


"""
Load data and create MultiAreaModel instance
"""
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)
SLN_completed = proc['SLN_completed']
SLN_Data_FV91 = proc['SLN_Data_FV91']
arch_types = proc['architecture_completed']

par = {'connection_params': {'g': -4.}}
M = MultiAreaModel(par, theory=True, simulation=False)

gain_matrix = M.K_matrix[:, :-1] * np.absolute(M.J_matrix[:, :-1])
eig = np.linalg.eig(gain_matrix)
gain_matrix_rescaled = gain_matrix / np.max(np.real(eig[0]))


# Create population-level graph and determine paths and path lengths
g = create_networkx_graph(gain_matrix_rescaled, M.structure_vec, relative=False)
paths, path_lengths = all_pairs_bellman_ford_path(g, weight='distance')

# Treat area MDP which does not receive connections from other areas
for area in area_list:
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
CC_paths = {area: {} for area in area_list}
for target_area in area_list:
    for source_area in area_list:
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

# Create area-level graph
K_area = np.zeros((32, 32))

for i, target in enumerate(area_list):
    for j, source in enumerate(area_list):
        K_area[i][j] = M.K_areas[target][source]

area_gain_matrix = K_area * np.ones_like(K_area) * M.J_matrix[0][0]
eig = np.linalg.eig(area_gain_matrix)
area_gain_matrix_rescaled = area_gain_matrix / np.max(np.real(eig[0]))

G = create_networkx_area_graph(area_gain_matrix_rescaled, area_list, relative=False)
gen = nx.all_pairs_dijkstra_path(G, weight='distance')
G_paths = {p[0]: p[1] for p in gen}
gen = nx.all_pairs_dijkstra_path_length(G, weight='distance')
G_path_lengths = {p[0]: p[1] for p in gen}
for area in G_paths:
    G_paths[area]['MDP'] = []
    G_path_lengths[area]['MDP'] = np.inf


"""
Figure layout.
"""


def layout_barplot_axes(ax):
    """
    Simple wrapper to set layout of the barplot axes.
    """
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")


ncols = 3
width = 6.8504
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

height = 5.25
print(width, height)
pl.rcParams['figure.figsize'] = (width, height)

fine_tune_cols = 2

fig = pl.figure()

axes = {}

panel_height = 0.12
panel_width = 0.27
panel_bottom = 0.173
axes['HL'] = pl.axes([0.03, panel_bottom, panel_width, panel_height])
axes['HZ'] = pl.axes([0.38, panel_bottom, panel_width, panel_height])
axes['LH'] = pl.axes([0.725, panel_bottom, panel_width, panel_height])

pl.text(0.02, 0.95, r'\bfseries{A}', fontdict={'fontsize': 10.,
                                               'weight': 'bold',
                                               'horizontalalignment': 'left',
                                               'verticalalignment': 'bottom'},
        transform=fig.transFigure)

pl.text(0.02, 0.63, r'\bfseries{B}', fontdict={'fontsize': 10.,
                                               'weight': 'bold',
                                               'horizontalalignment': 'left',
                                               'verticalalignment': 'bottom'},
        transform=fig.transFigure)

pl.text(0.02, 0.3, r'\bfseries{C}', fontdict={'fontsize': 10.,
                                              'weight': 'bold',
                                              'horizontalalignment': 'left',
                                              'verticalalignment': 'bottom'},
        transform=fig.transFigure)


"""
Analyze the paths between areas for different types of connections
and write out the linewidths for the corresponding tex plots to file.

1. Distinguish using SLN
"""
# Differential analysis of hierarchically directed connections
# FF = feedforward
# FB = feedback
FF_path_pairs = []
FB_path_pairs = []
lateral_path_pairs = []

FF_path_lengths = []
FB_path_lengths = []
lateral_path_lengths = []

FF_paths = []
FB_paths = []
lateral_paths = []

for target_area in area_list:
    for source_area in area_list:
        indices = create_mask(M.structure, target_pops=M.structure[target_area],
                              source_pops=M.structure[source_area],
                              target_areas=[target_area],
                              source_areas=[source_area],
                              complete_area_list=area_list,
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
        if (target_area != source_area and
            len(paths[source_pop][target_pop]) > 0 and source_area in
                SLN_completed[target_area]):
            if hierarchical_relation(target_area, source_area, SLN_completed) == 'FB':
                FB_path_pairs.append(pair)
                FB_paths.append(paths[source_pop][target_pop])
                FB_path_lengths.append(
                    path_lengths[source_pop][target_pop])
            elif hierarchical_relation(target_area, source_area, SLN_completed) == 'FF':
                FF_path_pairs.append(pair)
                FF_paths.append(paths[source_pop][target_pop])
                FF_path_lengths.append(
                    path_lengths[source_pop][target_pop])
            else:
                lateral_path_pairs.append(pair)
                lateral_paths.append(paths[source_pop][target_pop])
                lateral_path_lengths.append(
                    path_lengths[source_pop][target_pop])

# ## Statistics of path stages
FF_stages = [len(path) for path in FF_paths]
FB_stages = [len(path) for path in FB_paths]
lateral_stages = [len(path) for path in lateral_paths]

max_lw = 0.3  # This is an empirically determined value

C = Counter(FF_path_pairs)
fn = 'tex/lw_FF_paths_SLN.tex'
write_out_lw(fn, C)

C = Counter(FB_path_pairs)
fn = 'tex/lw_FB_paths_SLN.tex'
write_out_lw(fn, C)

C = Counter(lateral_path_pairs)
fn = 'tex/lw_lateral_paths_SLN.tex'
write_out_lw(fn, C)

"""
2. Use architectural types
"""
# HL = high to low type
# LH = low to high type
# HZ = equal arch. types
HL_path_pairs = []
LH_path_pairs = []
HZ_path_pairs = []

HL_path_lengths = []
LH_path_lengths = []
HZ_path_lengths = []

HL_paths = []
LH_paths = []
HZ_paths = []

for target_area in area_list:
    for source_area in area_list:
        indices = create_mask(M.structure, target_pops=M.structure[target_area],
                              source_pops=M.structure[source_area],
                              target_areas=[target_area],
                              source_areas=[source_area],
                              complete_area_list=area_list,
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

# ## Statistics of path stages
HL_stages = [len(path) for path in HL_paths]
LH_stages = [len(path) for path in LH_paths]
HZ_stages = [len(path) for path in HZ_paths]

C = Counter(HL_path_pairs)
fn = 'tex/lw_HL_paths.tex'
write_out_lw(fn, C)

C = Counter(LH_path_pairs)
fn = 'tex/lw_LH_paths.tex'
write_out_lw(fn, C)

C = Counter(HZ_path_pairs)
fn = 'tex/lw_HZ_paths.tex'
write_out_lw(fn, C)

C = Counter(HL_path_pairs)

"""
Analyze patterns of indirect paths
"""
HL_patterns = []
LH_patterns = []
HZ_patterns = []
l = [HL_patterns, LH_patterns, HZ_patterns]
l2 = [HL_paths, LH_paths, HZ_paths]

for patterns, path_list in zip(l, l2):
    for path in path_list:
        p = [(pop.split('-')[0], pop.split('-')[1]) for pop in path]
        s = ''
        area, pop = p.pop(0)
        s += pop
        while len(p) > 0:
            area2, pop2 = p.pop(0)
            if area == area2:
                s += '+' + pop2
            else:
                s += '-' + pop2
            area = area2
        patterns.append(s)

LH_pattern_count = Counter(LH_patterns)
HL_pattern_count = Counter(HL_patterns)
HZ_pattern_count = Counter(HZ_patterns)

LH_single_area_patterns = []
for p in LH_patterns:
    LH_single_area_patterns += p.split('-')[1:-1]

HL_single_area_patterns = []
for p in HL_patterns:
    HL_single_area_patterns += p.split('-')[1:-1]

HZ_single_area_patterns = []
for p in HZ_patterns:
    HZ_single_area_patterns += p.split('-')[1:-1]

# Plot values for HL paths
layout_barplot_axes(axes['HL'])

C = Counter(HL_single_area_patterns)
counts = list(C.values())
# Define order of pairs consistently across panels
pairs = ['5E', '23E', '6E']
counts = [C[p] for p in pairs]

axes['HL'].bar(list(range(len(counts))), counts,
               color=myblue, edgecolor='none')
axes['HL'].set_xticks([])
axes['HL'].set_yticks([])
axes['HL'].set_ylim((0., np.max(counts)))

# Plot values for HZ paths
layout_barplot_axes(axes['HZ'])

C = Counter(HZ_single_area_patterns)
counts = list(C.values())
# Define order of pairs consistently across panels
pairs = ['5E', '23E', '6E']
counts = [C[p] for p in pairs]

axes['HZ'].bar(list(range(len(counts))), counts,
               color=myblue, edgecolor='none')
axes['HZ'].set_xticks([])
axes['HZ'].set_yticks([])
axes['HZ'].set_ylim((0., np.max(counts)))

# Plot values for LH paths
layout_barplot_axes(axes['LH'])

C = Counter(LH_single_area_patterns)
counts = list(C.values())
# Define order of pairs consistently across panels
pairs = ['5E', '23E', '6E']
counts = [C[p] for p in pairs]

axes['LH'].bar(list(range(len(counts))), counts,
               color=myblue, edgecolor='none')
axes['LH'].set_xticks([])
axes['LH'].set_yticks([])
axes['LH'].set_ylim((0., np.max(counts)))


"""
Save figure and convert to eps
"""
pl.savefig('Fig8_laminar_paths_mpl.pdf')
os.system('pdftops -eps Fig8_laminar_paths_mpl.pdf')


"""
Create tex figures
"""
os.chdir('tex/')
os.system('bash compile_tex.sh')
os.chdir('../')


"""
Finally, merge the tex-created figures into the main figure.
"""
import pyx

c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0., 0., "Fig8_laminar_paths_mpl.eps", width=17.3))


c.insert(pyx.epsfile.epsfile(0.2, 9., "tex/FF_paths.eps", width=5.))
c.insert(pyx.epsfile.epsfile(6.2, 9., "tex/lateral_paths.eps", width=5.))
c.insert(pyx.epsfile.epsfile(12.2, 9., "tex/FB_paths.eps", width=5.))

c.insert(pyx.epsfile.epsfile(0.2, 4.8, "tex/HL_paths.eps", width=5.))
c.insert(pyx.epsfile.epsfile(6.2, 4.8, "tex/HZ_paths.eps", width=5.))
c.insert(pyx.epsfile.epsfile(12.2, 4.8, "tex/LH_paths.eps", width=5.))

c.insert(pyx.epsfile.epsfile(0.98, 0.8, "tex/indirect_5E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(2.48, 0.8, "tex/indirect_23E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(4.0, 0.8, "tex/indirect_6E.eps", width=0.75))

c.insert(pyx.epsfile.epsfile(7.02, 0.8, "tex/indirect_5E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(8.535, 0.8, "tex/indirect_23E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(10.075, 0.8, "tex/indirect_6E.eps", width=0.75))

c.insert(pyx.epsfile.epsfile(13.0, 0.8, "tex/indirect_5E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(14.5, 0.8, "tex/indirect_23E.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(16.05, 0.8, "tex/indirect_6E.eps", width=0.75))

c.writeEPSfile("Fig8_laminar_paths.eps")
