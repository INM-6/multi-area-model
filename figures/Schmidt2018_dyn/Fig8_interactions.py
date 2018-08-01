import csv
import copy
import json
import numpy as np
import os
import pyx

from helpers import original_data_path, infomap_path
from multiarea_model.multiarea_model import MultiAreaModel
from plotcolors import myred, myblue

import matplotlib.pyplot as pl
from matplotlib import gridspec
from matplotlib import rc_file
rc_file('plotstyle.rc')

import sys
sys.path.append('../Schmidt2018')
from graph_helpers import apply_map_equation

"""
Figure layout
"""
cmap = pl.cm.coolwarm
cmap = cmap.from_list('mycmap', [myblue, 'white', myred], N=256)
cmap2 = cmap.from_list('mycmap', ['white', myred], N=256)


width = 7.0866
n_horz_panels = 2.
n_vert_panels = 3.

fig = pl.figure()
axes = {}
gs1 = gridspec.GridSpec(1, 3)
gs1.update(left=0.05, right=0.95, top=0.95,
           bottom=0.52, wspace=0., hspace=0.4)
axes['A'] = pl.subplot(gs1[:, 0])
axes['B'] = pl.subplot(gs1[:, 1])
axes['C'] = pl.subplot(gs1[:, 2])

gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.18, right=0.8, top=0.35,
           wspace=0., bottom=0.13, hspace=0.2)
axes['D'] = pl.subplot(gs1[:, :])

gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.165, right=0.6, top=0.04,
           wspace=0., bottom=0.0, hspace=0.2)
axes['E'] = pl.subplot(gs1[:, :])

gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.688, right=0.95, top=0.04,
           wspace=0., bottom=0.0, hspace=0.2)
axes['F'] = pl.subplot(gs1[:, :])

for label in ['A', 'B', 'C', 'D', 'E', 'F']:
    if label in ['E', 'F']:
        label_pos = [-0.08, 1.01]
    else:
        label_pos = [-0.1, 1.01]
    pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
            fontdict={'fontsize': 16, 'weight': 'bold',
                      'horizontalalignment': 'left', 'verticalalignment':
                      'bottom'}, transform=axes[label].transAxes)
    axes[label].spines['right'].set_color('none')
    axes[label].spines['top'].set_color('none')
    axes[label].yaxis.set_ticks_position("left")
    axes[label].xaxis.set_ticks_position("bottom")

for label in ['E', 'F']:
    axes[label].spines['right'].set_color('none')
    axes[label].spines['top'].set_color('none')
    axes[label].spines['left'].set_color('none')
    axes[label].spines['bottom'].set_color('none')

    axes[label].yaxis.set_ticks_position("none")
    axes[label].xaxis.set_ticks_position("none")
    axes[label].set_yticks([])
    axes[label].set_xticks([])

"""
Load data
"""

"""
Create MultiAreaModel instance to have access to data structures
"""
conn_params = {'g': -11.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params}
M = MultiAreaModel(network_params)

# Load experimental functional connectivity
func_conn_data = {}
with open('Fig8_exp_func_conn.csv', 'r') as f:
    myreader = csv.reader(f, delimiter='\t')
    # Skip first 3 lines
    next(myreader)
    next(myreader)
    next(myreader)
    areas = next(myreader)
    for line in myreader:
        dict_ = {}
        for i in range(len(line)):
            dict_[areas[i]] = float(line[i])
        func_conn_data[areas[myreader.line_num - 5]] = dict_

exp_FC = np.zeros((len(M.area_list),
                   len(M.area_list)))
for i, area1 in enumerate(M.area_list):
    for j, area2 in enumerate(M.area_list):
        exp_FC[i][j] = func_conn_data[area1][area2]

fn = 'FC_exp_communities.json'
with open(fn, 'r') as f:
    part_exp = json.load(f)
part_exp_list = [part_exp[area] for area in M.area_list]


"""
Simulation data
"""
LOAD_ORIGINAL_DATA = True

cc_weights_factor = [1.0, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 2., 2.1, 2.5, 1.9]

if LOAD_ORIGINAL_DATA:
    labels = ['33fb5955558ba8bb15a3fdce49dfd914682ef3ea',
              '783cedb0ff27240133e3daa63f5d0b8d3c2e6b79',
              '380856f3b32f49c124345c08f5991090860bf9a3',
              '5a7c6c2d6d48a8b687b8c6853fb4d98048681045',
              'c1876856b1b2cf1346430cf14e8d6b0509914ca1',
              'a30f6fba65bad6d9062e8cc51f5483baf84a46b7',
              '1474e1884422b5b2096d3b7a20fd4bdf388af7e0',
              'f18158895a5d682db5002489d12d27d7a974146f',
              '08a3a1a88c19193b0af9d9d8f7a52344d1b17498',
              '5bdd72887b191ec22a5abcc04ca4a488ea216e32',
              '99c0024eacc275d13f719afd59357f7d12f02b77']
    data_path = original_data_path
    label_plot = labels[-1]  # chi=1.9
else:
    from network_simulations import init_models
    from config import data_path
    models = init_models('Fig8')
    labels = [M.simulation.label for M in models]


sim_FC = {}
for label in labels:
    fn = os.path.join(data_path,
                      label,
                      'Analysis',
                      'functional_connectivity_synaptic_input.npy')
    sim_FC[label] = np.load(fn)

sim_FC_bold = {}
for label in [label_plot]:
    fn = os.path.join(data_path,
                      label,
                      'Analysis',
                      'functional_connectivity_bold_signal.npy')
    sim_FC_bold[label] = np.load(fn)

label = label_plot
fn = os.path.join(data_path,
                  label,
                  'Analysis',
                  'FC_synaptic_input_communities.json')
with open(fn, 'r') as f:
    part_sim = json.load(f)
part_sim_list = [part_sim[area] for area in M.area_list]
part_sim_index = np.argsort(part_sim_list, kind='mergesort')
# Manually position MDP in between the two clusters for visual purposes
ind_MDP = M.area_list.index('MDP')
ind_MDP_index = np.where(part_sim_index == ind_MDP)[0][0]
part_sim_index = np.append(part_sim_index[:ind_MDP_index], part_sim_index[ind_MDP_index+1:])
new_ind_MDP_index = np.where(np.array(part_sim_list)[part_sim_index] == 0.)[0][-1]
part_sim_index = np.insert(part_sim_index, new_ind_MDP_index+1, ind_MDP)

    
def zero_diagonal(matrix):
    """
    Return copy of a matrix with diagonal set to zero.
    """
    M = copy.copy(matrix)
    for i in range(M.shape[0]):
        M[i, i] = 0
    return M


def matrix_plot(ax, matrix, index, vlim, pos=None):
    """
    Plot matrix into matplotlib axis sorted according to index.
    """
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    x = np.arange(0, len(M.area_list) + 1)
    y = np.arange(0, len(M.area_list[::-1]) + 1)
    X, Y = np.meshgrid(x, y)

    ax.set_xlim((0, 32))
    ax.set_ylim((0, 32))

    ax.set_aspect(1. / ax.get_data_ratio())

    vmax = vlim
    vmin = -vlim

    # , norm = LogNorm(1e-8,1.))
    im = ax.pcolormesh(matrix[index][:, index][::-1],
                       cmap=cmap, vmin=vmin, vmax=vmax)

    cbticks = [-1., -0.5, 0., 0.5, 1.0]
    cb = pl.colorbar(im, ax=ax, ticks=cbticks, fraction=0.046)
    cb.ax.tick_params(labelsize=14)
    ax.set_yticks([])

    if pos != (0, 2):
        cb.remove()
    else:
        ax.text(1.25, 0.52, r'FC', rotation=90,
                transform=ax.transAxes, size=14)
    ax.set_xticks([])

    ax.set_xlabel('Cortical area', size=14)
    ax.set_ylabel('Cortical area', size=14)

    
"""
Plotting
"""
ax = axes['A']
label = label_plot


matrix_plot(ax, zero_diagonal(sim_FC[label]),
            part_sim_index, 1., pos=(0, 0))

ax = axes['B']
label = label_plot

matrix_plot(ax, zero_diagonal(sim_FC_bold[label]),
            part_sim_index, 1., pos=(0, 0))

ax = axes['C']
matrix_plot(ax, zero_diagonal(exp_FC),
            part_sim_index, 1., pos=(0, 2))

areas = np.array(M.area_list)[part_sim_index]
area_string = areas[0]
for area in areas[1:]:
    area_string += ' '
    area_string += area

pl.text(0.02, 0.45, r'\bfseries{}Order of cortical areas', transform=fig.transFigure, fontsize=13)
pl.text(0.02, 0.42, area_string,
        transform=fig.transFigure, fontsize=13)


ax = axes['D']
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

cc_list = []
for i, label in enumerate(labels):
    cc = np.corrcoef(zero_diagonal(sim_FC[label]).flatten(),
                     zero_diagonal(exp_FC).flatten())[0][1]
    cc_list.append(cc)
    

ax.plot(cc_weights_factor[1:], cc_list[1:], '.', ms=10,
        markeredgecolor='none', label='Sim. vs. Exp.', color='k')
ax.plot(cc_weights_factor[0], cc_list[0], '^', ms=5,
        markeredgecolor='none', label='Sim. vs. Exp.', color='k')

label = label_plot
cc_bold = np.corrcoef(zero_diagonal(sim_FC_bold[label]).flatten(),
                      zero_diagonal(exp_FC).flatten())[0][1]
ax.plot([1.9], cc_bold, '.',
        ms=10, markeredgecolor='none', color=myred)

# Correlation between exp. FC and structural connectivity
# Construct the structural connectivity as the matrix of relative
conn_matrix = np.zeros((len(M.area_list), len(M.area_list)))
for i, area1 in enumerate(M.area_list):
    s = np.sum(list(M.K_areas[area1].values()))
    for j, area2 in enumerate(M.area_list):
        value = M.K_areas[area1][area2] / s
        conn_matrix[i][j] = value


cc = np.corrcoef(zero_diagonal(conn_matrix).flatten(),
                 zero_diagonal(exp_FC).flatten())[0][1]

# Formatting
ax.hlines(cc, -0.1, 2.5, linestyle='dashed', color='k')
ax.set_xlabel(r'Cortico-cortical weight factor $\chi$',
              labelpad=-0.1, size=16)
ax.set_ylabel(r'$r_{\mathrm{Pearson}}$', size=20)
ax.set_xlim((0.9, 2.7))
ax.set_ylim((-0.1, 0.6))
ax.set_yticks([0., 0.2, 0.4])
ax.set_yticklabels([0., 0.2, 0.4], size=13)
ax.set_xticks([1., 1.5, 2., 2.5])
ax.set_xticklabels([1., 1.5, 2., 2.5], size=13)


"""
Save figure
"""
pl.savefig('Fig8_interactions_mpl.eps')

"""
We compare the clusters found in the functional connectivity to
clusters found in the structural connectivity of the network. To
detect the clusters in the structural connectivity, we repeat the the
procedure from Fig. 7 of Schmidt et al. 'Multi-scale account of the
network structure of macaque visual cortex' and apply the map equation
method (see Materials & Methods in Schmidt et al. 2018) to the
structural connectivity of the network.

This requires installation of the infomap executable and defining the
path to the executable.
"""
fn = 'Fig8_structural_clusters'
modules, modules_areas, index = apply_map_equation(conn_matrix,
                                                   M.area_list,
                                                   filename=fn,
                                                   infomap_path=infomap_path)
with open('{}.map'.format(fn), 'r') as f:
    line = ''
    while '*Nodes' not in line:
        line = f.readline()
    line = f.readline()
    map_equation = []
    map_equation_areas = []
    while "*Links" not in line:
        map_equation.append(int(line.split(':')[0]))
        map_equation_areas.append(line.split('"')[1])
        line = f.readline()
    f.close()
    map_equation = np.array(map_equation)
    map_equation_dict = dict(
        list(zip(map_equation_areas, map_equation)))

# To create the alluvial input, we rename the simulated clusters
# 1S --> 2S, 2S ---> 1S for visual purposes
f = open('alluvial_input.txt', 'w')
f.write("area,map_equation, louvain, louvain_exp\n")
for i, area in enumerate(M.area_list):
    if part_sim_list[i] == 1:
        psm = 2
    elif part_sim_list[i] == 0:
        psm = 1
    s = '{}, {}, {}, {}'.format(area,
                                map_equation_dict[area],
                                psm,
                                part_exp_list[i])
    f.write(s)
    f.write('\n')
f.close()

# The alluvial plot cannot be created with a script. To reproduce the alluvial
# plot, go to http://app.rawgraphs.io/ and proceed from there.

"""
Merge with alluvial plot
"""
c = pyx.canvas.canvas()
c.fill(pyx.path.rect(0, 0., 17.9, 17.), [pyx.color.rgb.white])

c.insert(pyx.epsfile.epsfile(0., 6., "Fig8_interactions_mpl.eps", width=17.9))
c.insert(pyx.epsfile.epsfile(
    0.1, -1., "Fig8_alluvial_struct_sim.eps", width=11.))
c.insert(pyx.epsfile.epsfile(
    11.2, -0.6, "Fig8_alluvial_sim_exp.eps", width=6.5))

c.writeEPSfile("Fig8_interactions.eps")
