import numpy as np
import matplotlib.pyplot as pl
import os
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator

from M2E_compute_pop_rates import compute_pop_rates
from M2E_compute_rate_time_series import compute_rate_time_series

def plot_firing_rate(M):
    # load spike data and calculate instantaneous and mean firing rates
    data = np.loadtxt(M.simulation.data_dir + '/recordings/' + M.simulation.label + "-spikes-1-0.dat", skiprows=3)
    tsteps, spikecount = np.unique(data[:,1], return_counts=True)
    rate = spikecount / M.simulation.params['dt'] * 1e3 / np.sum(M.N_vec)
    
    # visualize calculate instantaneous and mean firing rates
    fig = pl.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.plot(tsteps, rate)
    
    # visualize the mean rate
    mean_rate = np.average(rate)
    ax.plot(tsteps, mean_rate*np.ones(len(tsteps)), label='mean')
    
    # display the value of mean rate
    ax.text(0.7 * max(tsteps), mean_rate+2, f'Mean firing rate: {mean_rate:.2f}', fontsize=12)

    ax.set_title('Instantaneous and mean firing rate of all populations', fontsize=15, pad=10)
    ax.set_xlabel('Time (ms)', fontsize=13)
    ax.set_ylabel('Firing rate (spikes / s)', fontsize=12)
    ax.set_xlim(0, M.simulation.params['t_sim'])
    ax.set_ylim(0, 50)
    ax.legend()
    pl.show()
    
    
def plot_firing_rate_over_areas(M, data_path):
    label = M.simulation.label
    area_list = M.simulation.params["areas_simulated"]
    
    # Compute pop_rates
    compute_pop_rates(M, data_path, label)
    
    # compute rate_time_series_full
    for area in area_list:
        compute_rate_time_series(M, data_path, label, area, 'full')
    
    # time series of firing rates
    rate_time_series = {}
    for area in area_list:
        fn = os.path.join(data_path, label,
                          'Analysis',
                          'rate_time_series_full',
                          'rate_time_series_full_{}.npy'.format(area))
        # fn = os.path.join(data_path, label,
        #                   'Analysis',
        #                   'rate_time_series-{}.npy'.format(area))
        rate_time_series[area] = np.load(fn)
        
    # print(rate_time_series)

    t_min = 500.
    t_max = M.simulation.params['t_sim']
    time = np.arange(t_min, t_max)
    matrix = []
    for area in area_list:
        # print(area)
        binned_spikes = rate_time_series[area][500:600]
        matrix.append(binned_spikes)
    
    matrix = np.array(matrix)
    normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
    fig = pl.figure()
    fig.suptitle('Normalized instantanous firing rate over simulated areas', fontsize=16, x=0.45, y=0.95)
    ax = pl.subplot()
    
    cmap = pl.get_cmap('YlOrBr')

    # masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    ax.patch.set_hatch('x')
    im = ax.pcolormesh(normalized_matrix, cmap=cmap, edgecolors='None')
    ax.set_xlim(0, normalized_matrix[0].size)
    
    ax.set_xticks([i for i in np.arange(0, 100, 10)])
    ax.set_xticklabels([i for i in np.arange(500, 600, 10)])
    ax.set_yticks([a + 0.5 for a in list(range(len(area_list)))])
    ax.set_yticklabels(area_list)
    ax.set_ylabel('Area', size=13)
    ax.set_xlabel('Time (ms)', size=13)
    # t = FixedLocator([0.01, 0.1, 1., 10., 100.])

    pl.colorbar(im)