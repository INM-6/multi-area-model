import json
import numpy as np
import os

import sys
sys.path.append('./figures/Schmidt2018_dyn')

# from helpers import original_data_path, population_labels
from helpers import population_labels
from multiarea_model import MultiAreaModel
from multiarea_model import Analysis
from plotcolors import myred, myblue

import matplotlib.pyplot as pl
from matplotlib import gridspec
# from matplotlib import rc_file
# rc_file('plotstyle.rc')

icolor = myred
ecolor = myblue

from M2E_LOAD_DATA import load_and_create_data

def set_boxplot_props(d):
    for i in range(len(d['boxes'])):
        if i % 2 == 0:
            d['boxes'][i].set_facecolor(icolor)
            d['boxes'][i].set_color(icolor)
        else:
            d['boxes'][i].set_facecolor(ecolor)
            d['boxes'][i].set_color(ecolor)
    pl.setp(d['whiskers'], color='k')
    pl.setp(d['fliers'], color='k', markerfacecolor='k', marker='+')
    pl.setp(d['medians'], color='none')
    pl.setp(d['caps'], color='k')
    pl.setp(d['means'], marker='x', color='k',
            markerfacecolor='k', markeredgecolor='k', markersize=3.)

def plot_resting_state(M, data_path, raster_areas=['V1', 'V2', 'FEF']):
    """
    Analysis class.
    An instance of the analysis class for the given network and simulation.
    Can be created as a member class of a multiarea_model instance or standalone.

    Parameters
    ----------
    network : MultiAreaModel
        An instance of the multiarea_model class that specifies
        the network to be analyzed.
    simulation : Simulation
        An instance of the simulation class that specifies
        the simulation to be analyzed.
    data_list : list of strings {'spikes', vm'}, optional
        Specifies which type of data is to load. Defaults to ['spikes'].
    load_areas : list of strings with area names, optional
        Specifies the areas for which data is to be loaded.
        Default value is None and leads to loading of data for all
        simulated areas.
    """
    # Instantiate an analysis class and load spike data
    A = Analysis(network=M, 
                 simulation=M.simulation, 
                 data_list=['spikes'],
                 load_areas=None)
    
    # load data
    load_and_create_data(M, A)
    
    t_sim = M.simulation.params["t_sim"]
    
    """
    Figure layout
    """

    nrows = 4
    ncols = 4
    # width = 7.0866
    width = 12
    panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

    height = width / panel_wh_ratio * float(nrows) / ncols
    pl.rcParams['figure.figsize'] = (width, height)


    fig = pl.figure()
    axes = {}

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(left=0.06, right=0.72, top=0.95, wspace=0.4, bottom=0.35)
    # axes['A'] = pl.subplot(gs1[:-1, :1])
    # axes['B'] = pl.subplot(gs1[:-1, 1:2])
    # axes['C'] = pl.subplot(gs1[:-1, 2:])
    axes['A'] = pl.subplot(gs1[:1, :1])
    axes['B'] = pl.subplot(gs1[:1, 1:2])
    axes['C'] = pl.subplot(gs1[:1, 2:])

    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(left=0.78, right=0.95, top=0.95, bottom=0.35)
    axes['D'] = pl.subplot(gs2[:1, :1])
    axes['E'] = pl.subplot(gs2[1:2, :1])
    axes['F'] = pl.subplot(gs2[2:3, :1])


    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(left=0.1, right=0.95, top=0.3, bottom=0.075)
    axes['G'] = pl.subplot(gs3[:1, :1])

    # areas = ['V1', 'V2', 'FEF']
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                 'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                 'STPa', '46', 'AITd', 'TH']
    if len(raster_areas) !=3:
        raise Exception("Error! Please give 3 areas to display as raster plots.")
    for area in raster_areas:
        if area not in area_list:
            raise Exception("Error! Given raster areas are either not from complete_area_list, please input correct areas to diaply the raster plots.")
    areas = raster_areas

    labels = ['A', 'B', 'C']
    for area, label in zip(areas, labels):
        label_pos = [-0.2, 1.01]
        # pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label + ': ' + area,
        #         fontdict={'fontsize': 10, 'weight': 'bold',
        #                   'horizontalalignment': 'left', 'verticalalignment':
        #                   'bottom'}, transform=axes[label].transAxes)
        pl.text(label_pos[0], label_pos[1], label + ': ' + area,
                 fontdict={'fontsize': 10, 'weight': 'bold', 
                           'horizontalalignment': 'left', 'verticalalignment': 
                           'bottom'}, transform=axes[label].transAxes)

    label = 'G'
    label_pos = [-0.1, 0.92]
    # pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
    #         fontdict={'fontsize': 10, 'weight': 'bold',
    #                   'horizontalalignment': 'left', 'verticalalignment':
    #                   'bottom'}, transform=axes[label].transAxes)
    pl.text(label_pos[0], label_pos[1], label,
             fontdict={'fontsize': 10, 'weight': 'bold', 
                       'horizontalalignment': 'left', 'verticalalignment': 
                       'bottom'}, transform=axes[label].transAxes)

    labels = ['E', 'D', 'F']
    for label in labels:
        label_pos = [-0.2, 1.05]
        # pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
        #         fontdict={'fontsize': 10, 'weight': 'bold',
        #                   'horizontalalignment': 'left', 'verticalalignment':
        #                   'bottom'}, transform=axes[label].transAxes)
        pl.text(label_pos[0], label_pos[1], label,
             fontdict={'fontsize': 10, 'weight': 'bold', 
                       'horizontalalignment': 'left', 'verticalalignment': 
                       'bottom'}, transform=axes[label].transAxes)
        

    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    for label in labels:
        axes[label].spines['right'].set_color('none')
        axes[label].spines['top'].set_color('none')
        axes[label].yaxis.set_ticks_position("left")
        axes[label].xaxis.set_ticks_position("bottom")

    for label in ['A', 'B', 'C']:
        axes[label].yaxis.set_ticks_position('none')


#     """
#     Load data
#     """
#     LOAD_ORIGINAL_DATA = True


#     if LOAD_ORIGINAL_DATA:
#         # use T=10500 simulation for spike raster plots
#         label_spikes = '3afaec94d650c637ef8419611c3f80b3cb3ff539'
#         # and T=100500 simulation for all other panels
#         label = '99c0024eacc275d13f719afd59357f7d12f02b77'
#         data_path = original_data_path
#     else:
#         from network_simulations import init_models
#         from config import data_path
#         models = init_models('Fig5')
#         label_spikes = models[0].simulation.label
#         label = models[1].simulation.label

    # """
    # Create MultiAreaModel instance to have access to data structures
    # """
    # M = MultiAreaModel({})

    # spike data
    # spike_data = {}
    # for area in areas:
    #     spike_data[area] = {}
    #     for pop in M.structure[area]:
    #         spike_data[area][pop] = np.load(os.path.join(data_path,
    #                                                      label_spikes,
    #                                                      'recordings',
    #                                                      '{}-spikes-{}-{}.npy'.format(label_spikes,
    #                                                                                   area, pop)))
    spike_data = A.spike_data
    label_spikes = M.simulation.label
    label = M.simulation.label
    
    # stationary firing rates
    fn = os.path.join(data_path, label, 'Analysis', 'pop_rates.json')
    with open(fn, 'r') as f:
        pop_rates = json.load(f)

    # time series of firing rates
    rate_time_series = {}
    for area in areas:
        # fn = os.path.join(data_path, label,
        #                   'Analysis',
        #                   'rate_time_series_full',
        #                   'rate_time_series_full_{}.npy'.format(area))
        fn = os.path.join(data_path, label,
                          'Analysis',
                          'rate_time_series-{}.npy'.format(area))
        rate_time_series[area] = np.load(fn)

    # time series of firing rates convolved with a kernel
    # rate_time_series_auto_kernel = {}
    # for area in areas:
    #     fn = os.path.join(data_path, label,
    #                       'Analysis',
    #                       'rate_time_series_auto_kernel',
    #                       'rate_time_series_auto_kernel_{}.npy'.format(area))
    #     rate_time_series_auto_kernel[area] = np.load(fn)

    # local variance revised (LvR)
    fn = os.path.join(data_path, label, 'Analysis', 'pop_LvR.json')
    with open(fn, 'r') as f:
        pop_LvR = json.load(f)

    # correlation coefficients
    # fn = os.path.join(data_path, label, 'Analysis', 'corrcoeff.json')
    fn = os.path.join(data_path, label, 'Analysis', 'synchrony.json')
    with open(fn, 'r') as f:
        corrcoeff = json.load(f)

    """
    Plotting
    """
    # print("Raster plots")

    # t_min = 3000.
    # t_max = 3500.
    t_min = t_sim - 500
    t_max = t_sim

    icolor = myred
    ecolor = myblue

    # frac_neurons = 0.03
    frac_neurons = 1

    for i, area in enumerate(areas):
        ax = axes[labels[i]]

        if area in spike_data:
            n_pops = len(spike_data[area])
            # Determine number of neurons that will be plotted for this area (for
            # vertical offset)
            offset = 0
            n_to_plot = {}
            for pop in M.structure[area]:
                n_to_plot[pop] = int(M.N[area][pop] * frac_neurons)
                offset = offset + n_to_plot[pop]
            y_max = offset + 1
            prev_pop = ''
            yticks = []
            yticklocs = []
            for jj, pop in enumerate(M.structure[area]):
                if pop[0:-1] != prev_pop:
                    prev_pop = pop[0:-1]
                    yticks.append('L' + population_labels[jj][0:-1])
                    yticklocs.append(offset - 0.5 * n_to_plot[pop])
                ind = np.where(np.logical_and(
                    spike_data[area][pop][:, 1] <= t_max, spike_data[area][pop][:, 1] >= t_min))
                pop_data = spike_data[area][pop][ind]
                pop_neurons = np.unique(pop_data[:, 0])
                neurons_to_ = np.arange(np.min(spike_data[area][pop][:, 0]), np.min(
                    spike_data[area][pop][:, 0]) + n_to_plot[pop], 1)

                if pop.find('E') > (-1):
                    pcolor = ecolor
                else:
                    pcolor = icolor

                for kk in range(n_to_plot[pop]):
                    spike_times = pop_data[pop_data[:, 0] == neurons_to_[kk], 1]

                    _ = ax.plot(spike_times, np.zeros(len(spike_times)) +
                                offset - kk, '.', color=pcolor, markersize=1)
                offset = offset - n_to_plot[pop]
            y_min = offset
            ax.set_xlim([t_min, t_max])
            ax.set_ylim([y_min, y_max])
            ax.set_yticklabels(yticks)
            ax.set_yticks(yticklocs)
            ax.set_xlabel('Time (s)', labelpad=-0.1)
            ax.set_xticks([t_min, t_min + 250., t_max])
            # ax.set_xticklabels([r'$3.$', r'$3.25$', r'$3.5$'])
            l = t_min/1000
            m = (t_min + t_max)/2000
            r = t_max/1000
            ax.set_xticklabels([f'{l:.2f}', f'{m:.2f}', f'{r:.2f}'])

    # print("plotting Population rates")

    rates = np.zeros((len(M.area_list), 8))
    for i, area in enumerate(M.area_list):
        for j, pop in enumerate(M.structure[area][::-1]):
            rate = pop_rates[area][pop][0]
            if rate == 0.0:
                rate = 1e-5
            if area == 'TH' and j > 3:  # To account for missing layer 4 in TH
                rates[i][j + 2] = rate
            else:
                rates[i][j] = rate


    rates = np.transpose(rates)
    masked_rates = np.ma.masked_where(rates < 1e-4, rates)

    ax = axes['D']
    d = ax.boxplot(np.transpose(rates), vert=False,
                   patch_artist=True, whis=1.5, showmeans=True)
    set_boxplot_props(d)

    ax.plot(np.mean(rates, axis=1), np.arange(
        1., len(M.structure['V1']) + 1., 1.), 'x', color='k', markersize=3)
    ax.set_yticklabels(population_labels[::-1], size=8)
    ax.set_yticks(np.arange(1., len(M.structure['V1']) + 1., 1.))
    ax.set_ylim((0., len(M.structure['V1']) + .5))

    x_max = 220.
    ax.set_xlim((-1., x_max))
    ax.set_xlabel(r'Rate (spikes/s)', labelpad=-0.1)
    ax.set_xticks([0., 50., 100.])

    # print("plotting Synchrony")
    
    syn = np.zeros((len(M.area_list), 8))
    for i, area in enumerate(M.area_list):
        for j, pop in enumerate(M.structure[area][::-1]):
            value = corrcoeff[area][pop]
            if value == 0.0:
                value = 1e-5
            if area == 'TH' and j > 3:  # To account for missing layer 4 in TH
                syn[i][j + 2] = value
            else:
                syn[i][j] = value


    syn = np.transpose(syn)
    masked_syn = np.ma.masked_where(syn < 1e-4, syn)

    ax = axes['E']
    d = ax.boxplot(np.transpose(syn), vert=False,
                   patch_artist=True, whis=1.5, showmeans=True)
    set_boxplot_props(d)

    ax.plot(np.mean(syn, axis=1), np.arange(
        1., len(M.structure['V1']) + 1., 1.), 'x', color='k', markersize=3)

    ax.set_yticklabels(population_labels[::-1], size=8)
    ax.set_yticks(np.arange(1., len(M.structure['V1']) + 1., 1.))
    ax.set_ylim((0., len(M.structure['V1']) + .5))
    ax.set_xticks(np.arange(0.0, 0.601, 0.2))
    ax.set_xlabel('Correlation coefficient', labelpad=-0.1)


    # print("plotting Irregularity")

    LvR = np.zeros((len(M.area_list), 8))
    for i, area in enumerate(M.area_list):
        for j, pop in enumerate(M.structure[area][::-1]):
            value = pop_LvR[area][pop]
            if value == 0.0:
                value = 1e-5
            if area == 'TH' and j > 3:  # To account for missing layer 4 in TH
                LvR[i][j + 2] = value
            else:
                LvR[i][j] = value

    LvR = np.transpose(LvR)
    masked_LvR = np.ma.masked_where(LvR < 1e-4, LvR)

    ax = axes['F']
    d = ax.boxplot(np.transpose(LvR), vert=False,
                   patch_artist=True, whis=1.5, showmeans=True)
    set_boxplot_props(d)

    ax.plot(np.mean(LvR, axis=1), np.arange(
        1., len(M.structure['V1']) + 1., 1.), 'x', color='k', markersize=3)
    ax.set_yticklabels(population_labels[::-1], size=8)
    ax.set_yticks(np.arange(1., len(M.structure['V1']) + 1., 1.))
    ax.set_ylim((0., len(M.structure['V1']) + .5))


    x_max = 2.9
    ax.set_xlim((0., x_max))
    ax.set_xlabel('Irregularity', labelpad=-0.1)
    ax.set_xticks([0., 1., 2.])

    axes['G'].spines['right'].set_color('none')
    axes['G'].spines['left'].set_color('none')
    axes['G'].spines['top'].set_color('none')
    axes['G'].spines['bottom'].set_color('none')
    axes['G'].yaxis.set_ticks_position("none")
    axes['G'].xaxis.set_ticks_position("none")
    axes['G'].set_xticks([])
    axes['G'].set_yticks([])


    # print("Plotting rate time series")
    pos = axes['G'].get_position()
    ax = []
    h = pos.y1 - pos.y0
    w = pos.x1 - pos.x0
    ax.append(pl.axes([pos.x0, pos.y0, w, 0.28 * h]))
    ax.append(pl.axes([pos.x0, pos.y0 + 0.33 * h, w, 0.28 * h]))
    ax.append(pl.axes([pos.x0, pos.y0 + 0.67 * h, w, 0.28 * h]))

    colors = ['0.5', '0.3', '0.0']

    # t_min = 500.
    # t_max = 10500.
    t_min = 500.
    t_max = t_sim
    # time = np.arange(500., t_max)
    time = np.arange(500, t_max)
    for i, area in enumerate(areas[::-1]):
        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')
        ax[i].yaxis.set_ticks_position("left")
        ax[i].xaxis.set_ticks_position("none")

        binned_spikes = rate_time_series[area][np.where(
            np.logical_and(time >= t_min, time < t_max))]
        ax[i].plot(time, binned_spikes, color=colors[0], label=area)
        # rate = rate_time_series_auto_kernel[area]
        rate = rate_time_series[area]
        ax[i].plot(time, rate, color=colors[2], label=area)
        ax[i].set_xlim((t_min, t_max))

        ax[i].text(0.8, 0.7, area, transform=ax[i].transAxes)

        if i > 0:
            ax[i].spines['bottom'].set_color('none')
            ax[i].set_xticks([])
            ax[i].set_yticks([0., 30.])
        else:
            # ax[i].set_xticks([1000., 5000., 10000.])
            ax[i].set_xticks([t_min, (t_min+t_max)/2, t_max])
            l = t_min/1000
            m = (t_min + t_max)/2000
            r = t_max/1000
            # ax[i].set_xticklabels([r'$1.$', r'$5.$', r'$10.$'])
            ax[i].set_xticklabels([f'{l:.2f}', f'{m:.2f}', f'{r:.2f}'])
            ax[i].set_yticks([0., 5.])
        if i == 1:
            ax[i].set_ylabel(r'Rate (spikes/s)')

    ax[0].set_xlabel('Time (s)', labelpad=-0.05)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95,
                        bottom=0.075, wspace=1., hspace=.5)

    # pl.savefig('Fig5_ground_state.eps')