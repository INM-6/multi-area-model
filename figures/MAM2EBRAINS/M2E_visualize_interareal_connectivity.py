import numpy as np
import matplotlib.pyplot as pl

import sys
sys.path.append('./figures/Schmidt2018')

from helpers import area_list, datapath
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
from multiarea_model import MultiAreaModel

def visualize_interareal_connectivity(M):
    """
    Visualize inter-area connectivity for a comparison of the full-scale model and the downscaled model
    
    Parameters:
        - M ((MultiAreaModel)): Object containing simulation data.
        
    Returns:
        None
    """
    # Full-scale model
    M_full_scale = MultiAreaModel({})
    
    """
    Figure layout
    """
    nrows = 1
    ncols = 2
    width = 12
    panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

    height = width / panel_wh_ratio * float(nrows) / ncols
    pl.rcParams['figure.figsize'] = (width, height)

    fig = pl.figure()
    fig.suptitle('Area-level connectivity of the full-scale and downscaled MAM expressed as relative indegrees for each target area', fontsize=15, x=0.5, y=1.05)
    axes = {}

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    
    axes['B'] = pl.subplot(gs1[:1, :1])
    axes['D'] = pl.subplot(gs1[:1, 1:2])

    pos2 = axes['D'].get_position()

    labels = ['B', 'D']
    labels_display = ['Full-scale model', 'Downscaled model']
    for i in range(len(labels)):
        label = labels[i]
        label_display = labels_display[i]
        if label in ['C']:
            label_pos = [-0.045, 1.18]
        else:
            label_pos = [-0.2, 1.04]
        pl.text(label_pos[0], label_pos[1], label_display,
                 fontdict={'fontsize': 12, 'weight': 'bold', 
                           'horizontalalignment': 'left', 'verticalalignment': 
                           'bottom'}, transform=axes[label].transAxes)

    """
    Panel B: Interareal connectivity of full-scaling multi-area model
    """
    conn_matrix_full_scale = np.zeros((32, 32))
    for i, area1 in enumerate(area_list[::-1]):
        for j, area2 in enumerate(area_list):
            conn_matrix_full_scale[i][j] = M_full_scale.K_areas[area1][
                area2] / np.sum(list(M_full_scale.K_areas[area1].values()))

    ax = axes['B']
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")

    ax.set_aspect(1. / ax.get_data_ratio())

    masked_matrix_full_scale = np.ma.masked_values(conn_matrix_full_scale, 0.0)

    cmap = pl.get_cmap('YlOrBr')
    cmap.set_bad('w', 1.0)

    x = np.arange(0, len(area_list) + 1)
    y = np.arange(0, len(area_list[::-1]) + 1)
    X, Y = np.meshgrid(x, y)

    ax.set_xticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    ax.set_xticklabels(area_list, rotation=90, size=10.)

    ax.set_yticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    ax.set_yticklabels(area_list[::-1], size=10.)

    ax.set_ylabel('Target area', fontsize=15)
    ax.set_xlabel('Source area', fontsize=15)
    im = ax.pcolormesh(masked_matrix_full_scale, cmap=cmap,
                       edgecolors='None', norm=LogNorm(vmin=1e-6, vmax=1.))

    t = FixedLocator([1e-6, 1e-4, 1e-2, 1])
    cbar = pl.colorbar(im, ticks=t, fraction=0.046, ax=ax)
    cbar.set_alpha(0.)

    """
    Panel D: Interareal connectivity of downscaling multi-area model
    """
    conn_matrix_down_scale = np.zeros((32, 32))
    for i, area1 in enumerate(area_list[::-1]):
        for j, area2 in enumerate(area_list):
            conn_matrix_down_scale[i][j] = M.K_areas[area1][
                area2] / np.sum(list(M.K_areas[area1].values()))
    
    ax = axes['D']
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")

    ax.set_aspect(1. / ax.get_data_ratio())

    masked_matrix_down_scale = np.ma.masked_values(conn_matrix_down_scale, 0.0)
    cmap.set_bad('w', 1.0)

    x = np.arange(0, len(area_list) + 1)
    y = np.arange(0, len(area_list[::-1]) + 1)
    X, Y = np.meshgrid(x, y)

    ax.set_xticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    ax.set_xticklabels(area_list, rotation=90, size=10.)

    ax.set_yticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    ax.set_yticklabels(area_list[::-1], size=10.)

    ax.set_ylabel('Target area', fontsize=15)
    ax.set_xlabel('Source area', fontsize=15)
    im = ax.pcolormesh(masked_matrix_down_scale, cmap=cmap,
                       edgecolors='None', norm=LogNorm(vmin=1e-6, vmax=1.))

    t = FixedLocator([1e-6, 1e-4, 1e-2, 1])
    cbar = pl.colorbar(im, ticks=t, fraction=0.046, ax=ax)
    cbar.set_alpha(0.)