import numpy as np

from multiarea_model import Analysis
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator

def plot_time_averaged_population_rates(M, area_list=None, **keywords):
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
    
    A = Analysis(network=M, 
                 simulation=M.simulation, 
                 data_list=['spikes'],
                 load_areas=None)
    
    A.create_pop_rates()
    
    if area_list is None:
        area_list = ['V1', 'V2', 'VP', 'V3', 'PIP', 'V3A', 'MT', 'V4t', 'V4',
                     'PO', 'VOT', 'DP', 'MIP', 'MDP', 'MSTd', 'VIP', 'LIP',
                     'PITv', 'PITd', 'AITv', 'MSTl', 'FST', 'CITv', 'CITd',
                     '7a', 'STPp', 'STPa', 'FEF', '46', 'TF', 'TH', 'AITd']

    matrix = np.zeros((len(area_list), len(A.network.structure['V1'])))

    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Time-averaged population rates encoded in colors', fontsize=15, loc='center')
    ax = fig.add_subplot(111)

    for i, area in enumerate(area_list):
        # print(i, area)
        # for j, pop in enumerate(A.network.structure_reversed['V1']):
        for j, pop in enumerate(A.network.structure['V1'][::-1]):
            if pop in A.network.structure[area]:
                rate = A.pop_rates[area][pop][0]
                if rate == 0.0:
                    rate = 1e-5  # To distinguish zero-rate from non-existing populations
            else:
                rate = np.nan
            matrix[i][j] = rate

    cm = plt.cm.jet
    cm = cm.from_list('mycmap', [(0., 64./255., 192./255.),  # custom dark blue
                                 (0., 128./255., 192./255.),  # custom light blue
                                 'white',
                                 (245./255., 157./255., 115./255.),  # custom light red
                                 (192./255., 64./255., 0.)], N=256)  # custom dark red
    cm.set_under('0.3')
    cm.set_bad('k')

    matrix = np.transpose(matrix)
    masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    ax.patch.set_hatch('x')
    im = ax.pcolormesh(masked_matrix, cmap=cm, edgecolors='None', norm=LogNorm(
        vmin=0.01, vmax=100.))
    ax.set_xlim(0, matrix[0].size)

    x_index = np.arange(4.5, 31.6, 5.0)
    x_ticks = [int(a + 0.5) for a in x_index]
    y_index = list(range(len(A.network.structure['V1'])))
    y_index = [a + 0.5 for a in y_index]
    # print(A.network.structure['V1'])
    # ax.set_xticks(x_index)
    ax.set_xticks([i + 0.5 for i in np.arange(0, len(area_list), 1)])
    # ax.set_xticklabels(x_ticks)
    ax.set_xticklabels(area_list, rotation=90, size=10.)
    ax.set_yticks(y_index)
    # ax.set_yticklabels(A.network.structure_reversed['V1'])
    ax.set_yticklabels(A.network.structure['V1'][::-1])
    ax.set_ylabel('Population', size=13)
    ax.set_xlabel('Area index', size=13)
    t = FixedLocator([0.01, 0.1, 1., 10., 100.])

    plt.colorbar(im, ticks=t)

    if 'output' in keywords:
        plt.savefig(os.path.join(A.output_dir, '{}_rates.{}'.format(A.simulation.label,
                                                                       keywords['output'])))
    else:
        fig.show()