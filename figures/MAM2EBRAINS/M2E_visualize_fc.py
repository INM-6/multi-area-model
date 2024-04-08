import csv
import copy
import json
import numpy as np
import os

import sys
sys.path.append('./figures/Schmidt2018_dyn')

from plotcolors import myred, myblue

import matplotlib.pyplot as pl
from matplotlib import gridspec

sys.path.append('./figures/Schmidt2018')

from M2E_compute_fc import compute_fc

cmap = pl.cm.coolwarm
cmap = cmap.from_list('mycmap', [myblue, 'white', myred], N=256)
cmap2 = cmap.from_list('mycmap', ['white', myred], N=256)

def zero_diagonal(matrix):
    """
    Return copy of a matrix with diagonal set to zero.
    
    Parameters:
        - matrix (ndarray): Matrix to copy.
        
    Returns:
        - M (ndarray): Matrix with diagonal set to zero.
    """
    M = copy.copy(matrix)
    for i in range(M.shape[0]):
        M[i, i] = 0
    return M


def matrix_plot(M, ax, matrix, index, vlim, pos=None):
    """
    Plot matrix into matplotlib axis sorted according to index.
    
    Parameters:
        - M (MultiAreaModel): Object containing simulation data.
        - ax (matplotlib axis): Axis to plot matrix into.
        - matrix (ndarray): Matrix to plot.
        - index (int): Index of matrix to plot.
        - vlim (float): Value limit.
        - pos (tuple): Position of matrix in figure.
        
    Returns:
        None
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

    im = ax.pcolormesh(matrix[index][:, index][::-1],
                       cmap=cmap, vmin=vmin, vmax=vmax)

    cbticks = [-1., -0.5, 0., 0.5, 1.0]
    cb = pl.colorbar(im, ax=ax, ticks=cbticks, fraction=0.046)
    cb.ax.tick_params(labelsize=14)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('Cortical area', size=14)
    ax.set_ylabel('Cortical area', size=14)
    
    
def visualize_fc(M, data_path):
    """
    Visualize functional connectivity.
    
    Parameters:
        - M ((MultiAreaModel)): Object containing simulation data.
        - data_path (str): Path to the data directory
        
    Returns:
        None
    """
    label = M.simulation.label
    
    # Compute functional connectivity
    compute_fc(M, data_path, label)
    
    """
    Figure layout
    """
    fig = pl.figure(figsize=(9, 4))
    fig.suptitle('Simulated functional connectivity (left) and FC of macaque resting-state fMRI', 
                 fontsize=17, x=0.53, y=1.15)
    axes = {}
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(left=0.05, right=0.95, top=1,
               bottom=0.3, wspace=0.3, hspace=0)
    axes['A'] = pl.subplot(gs1[:1, :1])
    axes['B'] = pl.subplot(gs1[:1, 1:2])

    for label in ['A', 'B']:
        if label in ['E', 'F']:
            label_pos = [-0.08, 1.01]
        else:
            label_pos = [-0.1, 1.01]
        pl.text(label_pos[0], label_pos[1], label,
                fontdict={'fontsize': 16, 'weight': 'bold',
                          'horizontalalignment': 'left', 'verticalalignment':
                          'bottom'}, transform=axes[label].transAxes)
        axes[label].spines['right'].set_color('none')
        axes[label].spines['top'].set_color('none')
        axes[label].yaxis.set_ticks_position("left")
        axes[label].xaxis.set_ticks_position("bottom")

    """
    Load data
    """
    # Load experimental functional connectivity
    func_conn_data = {}
    with open('./figures/Schmidt2018_dyn/Fig8_exp_func_conn.csv', 'r') as f:
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

    """
    Simulation data
    """
    labels = [M.simulation.label]

    sim_FC = {}
    for label in labels:
        fn = os.path.join(data_path,
                          label,
                          'Analysis',
                          'functional_connectivity_synaptic_input.npy')
        sim_FC[label] = np.load(fn)

    label = M.simulation.label
    areas_reordered = ['V1', 'V2', 'VP', 'V4t', 'V4', 'VOT', 'MSTd', 'PITv', 
                   'PITd', 'CITv', 'CITd', 'AITv', 'AITd', 'MDP', 'V3', 'V3A',
                   'MT', 'PIP', 'PO', 'DP', 'MIP', 'VIP', 'LIP', 'MSTI', 
                   'FEF', 'TF', 'FST', '7a', 'STPp', 'STPa', '46', 'TH']

    part_sim = {area: M.area_list.index(area) for area in areas_reordered if area in M.area_list}
    part_sim_index = list(part_sim.values())

    """
    Plotting
    """
    ax = axes['A']

    matrix_plot(M, ax, zero_diagonal(sim_FC[label]),
                part_sim_index, 1., pos=(0, 0))

    ax = axes['B']
    matrix_plot(M, ax, zero_diagonal(exp_FC),
                part_sim_index, 1., pos=(0, 0))
        
    areas = np.array(M.area_list)[part_sim_index]
    area_string = areas[0]
    for area in areas[1:]:
        area_string += ' '
        area_string += area

    pl.text(0.00, 0.15, r'Order of cortical areas:', transform=fig.transFigure, fontsize=13, fontweight='bold')
    pl.text(0.00, 0.1, area_string,
        transform=fig.transFigure, fontsize=11)
