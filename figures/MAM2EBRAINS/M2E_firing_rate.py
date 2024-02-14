import numpy as np
import matplotlib.pyplot as pl
import os
import json
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator

from M2E_compute_pop_rates import compute_pop_rates
from M2E_compute_rate_time_series import compute_rate_time_series

# Function for computing and printing the mean firing rate for all and only all simulated populations
def mean_firing_rate(M, data_path):
    label = M.simulation.label
    
    # Compute firing rate for all simulated populations
    compute_pop_rates(M, data_path, label)
    
    # Load the pop_rates data
    fn = os.path.join(data_path, label, 'Analysis', 'pop_rates.json')
    with open(fn, 'r') as f:
        pop_rates = json.load(f)
    
    # Calculate mean firing rate over all simulated populations
    rates = np.zeros((len(M.area_list), 8))
    for i, area in enumerate(M.area_list):
        for j, pop in enumerate(M.structure[area][::-1]):
            # rate = pop_rates[area][pop][0]
            rate = pop_rates[area][pop]
            if rate == 0.0:
                rate = 1e-5
            if area == 'TH' and j > 3:  # To account for missing layer 4 in TH
                rates[i][j + 2] = rate
            else:
                rates[i][j] = rate

    rates = np.transpose(rates)
    mfr = np.mean(np.mean(rates, axis=1))
    
    print("The mean firing rate over all simulated populations is {0:.3f} spikes/s.".format(mfr))

# Function for visualizing the instantaneous firing rate over all simulated areas
def plot_firing_rate_over_areas(M, data_path):
    area_list = M.simulation.params["areas_simulated"]
    label = M.simulation.label
    
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
    fig.suptitle('Normalized instantaneous firing rate over simulated areas', fontsize=16, x=0.45, y=0.95)
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