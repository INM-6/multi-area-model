import numpy as np
import os
import json

from multiarea_model import Analysis
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator

def plot_time_averaged_population_rates(M, data_path, area_list=None, **keywords):
    """
    Plot overview over time-averaged population rates encoded in colors
    with areas along x-axis and populations along y-axis.

    Parameters
    ----------
    area_list : list, optional
       Specifies with areas are plotted in which order.
       Default to None, leading to plotting of  all areas ordered by architectural type.
    output : {'pdf', 'png', 'eps'}, optional
        If given, the function stores the plot to a file of the given format.
    """

    # with open(os.path.join(data_path, M.simulation.label, 'custom_params_{}'.format(M.simulation.label)), 'r') as f:
    #     sim_params = json.load(f)
    # areas_simulated = sim_params['sim_params']['areas_simulated']
    area_list = M.simulation.params["areas_simulated"]
    
    # matrix = np.zeros((len(area_list), len(A.network.structure['V1'])))
    matrix = np.zeros((len(area_list), len(M.structure['V1'])))

    fig = plt.figure(figsize=(12, 3))
    fig.suptitle('Time-averaged firing rate over simulated populations', fontsize=15, x=0.43)
    ax = fig.add_subplot(111)
    
    # stationary firing rates
    fn = os.path.join(data_path, M.simulation.label, 'Analysis', 'pop_rates.json')
    with open(fn, 'r') as f:
        pop_rates = json.load(f)
    
    for i, area in enumerate(area_list):
        # print(i, area)
        # for j, pop in enumerate(A.network.structure_reversed['V1']):
        for j, pop in enumerate(M.structure['V1'][::-1]):
            if pop in M.structure[area]:
                # rate = A.pop_rates[area][pop][0]
                rate = pop_rates[area][pop]
                if rate == 0.0:
                    rate = 1e-5  # To distinguish zero-rate from non-existing populations
            else:
                rate = np.nan
            matrix[i][j] = rate

    # cm = plt.cm.jet
    # cm = cm.from_list('mycmap', [(0., 64./255., 192./255.),  # custom dark blue
    #                              (0., 128./255., 192./255.),  # custom light blue
    #                              'white',
    #                              (245./255., 157./255., 115./255.),  # custom light red
    #                              (192./255., 64./255., 0.)], N=256)  # custom dark red
    cm = plt.get_cmap('YlOrBr')
    cm.set_under('0.3')
    # cm.set_bad('k')
    cm.set_bad('white')

    matrix = np.transpose(matrix)
    masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    ax.patch.set_hatch('x')
    # im = ax.pcolormesh(masked_matrix, cmap=cm, edgecolors='None', norm=LogNorm(
    #     vmin=0.01, vmax=100.))
    im = ax.pcolormesh(masked_matrix, cmap=cm, edgecolors='None', vmin=0)
    ax.set_xlim(0, matrix[0].size)

    x_index = np.arange(4.5, 31.6, 5.0)
    x_ticks = [int(a + 0.5) for a in x_index]
    # y_index = list(range(len(A.network.structure['V1'])))
    y_index = list(range(len(M.structure['V1'])))
    y_index = [a + 0.5 for a in y_index]
    # ax.set_xticks(x_index)
    ax.set_xticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    # ax.set_xticklabels(x_ticks)
    ax.set_xticklabels(area_list, rotation=90, size=10.)
    ax.set_yticks(y_index)
    # ax.set_yticklabels(A.network.structure_reversed['V1'])
    # ax.set_yticklabels(M.structure['V1'][::-1])
    ax.set_yticklabels(['6I', '6E', '5I', '5E', '4I', '4E', '2/3I', '2/3E'])
    ax.set_ylabel('Population', size=13)
    ax.set_xlabel('Area index', size=13)
    # t = FixedLocator([0.01, 0.1, 1., 10., 100.])
    # t = FixedLocator([0, 10, 20, 30, 40, 50])
    
    # Iterate over the data and add 'X' for masked cells
    for i in range(masked_matrix.shape[0]):
        for j in range(masked_matrix.shape[1]):
            if masked_matrix.mask[i, j]:
                ax.text(j + 0.5, i + 0.5, 'X', va='center', ha='center', color='black', fontsize=23)

    # plt.colorbar(im, ticks=t)
    plt.colorbar(im)

    if 'output' in keywords:
        # plt.savefig(os.path.join(A.output_dir, '{}_rates.{}'.format(A.simulation.label,
        #                                                                keywords['output'])))
        plt.savefig(os.path.join(M.output_dir, '{}_rates.{}'.format(M.simulation.label,
                                                                       keywords['output'])))
    else:
        fig.show()