# Time-averaged population rates
# An overview over time-averaged population rates encoded in colors with areas along x-axis and populations along y-axis.

# def plot_time_averaged_population_rates(M, A):
#     A.show_rates()
    
def plot_time_averaged_population_rates(M):
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
    area_list = None
    if area_list is None:
        area_list = ['V1', 'V2', 'VP', 'V3', 'PIP', 'V3A', 'MT', 'V4t', 'V4',
                     'PO', 'VOT', 'DP', 'MIP', 'MDP', 'MSTd', 'VIP', 'LIP',
                     'PITv', 'PITd', 'AITv', 'MSTl', 'FST', 'CITv', 'CITd',
                     '7a', 'STPp', 'STPa', 'FEF', '46', 'TF', 'TH', 'AITd']

    matrix = np.zeros((len(area_list), len(M.network.structure['V1'])))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    for i, area in enumerate(area_list):
        # print(i, area)
        # for j, pop in enumerate(M.network.structure_reversed['V1']):
        for j, pop in enumerate(M.network.structure['V1']):
            if pop in M.network.structure[area]:
                rate = M.pop_rates[area][pop][0]
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
    y_index = list(range(len(M.network.structure['V1'])))
    y_index = [a + 0.5 for a in y_index]
    # print(M.network.structure['V1'])
    ax.set_xticks(x_index)
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(y_index)
    # ax.set_yticklabels(M.network.structure_reversed['V1'])
    ax.set_yticklabels(M.network.structure['V1'])
    ax.set_ylabel('Population', size=18)
    ax.set_xlabel('Area index', size=18)
    t = FixedLocator([0.01, 0.1, 1., 10., 100.])

    plt.colorbar(im, ticks=t)

    if 'output' in keywords:
        plt.savefig(os.path.join(M.output_dir, '{}_rates.{}'.format(M.simulation.label,
                                                                       keywords['output'])))
    else:
        fig.show()
