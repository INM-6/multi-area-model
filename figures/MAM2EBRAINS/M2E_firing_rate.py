import numpy as np
import matplotlib.pyplot as pl
import os
import json

from M2E_compute_pop_rates import compute_pop_rates
from M2E_compute_rate_time_series import compute_rate_time_series

def mean_firing_rate(M, data_path):
    """
    Calculate the mean firing rate for all simulated populations.

    Parameters:
        - M (MultiAreaModel): The simulation object.
        - data_path (str): The path to the data directory.

    Returns:
        None
    """
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

def plot_firing_rate_over_areas(M, data_path):
    """
    Generate a plot of firing rate over different areas based on the provided simulation data.

    Parameters:
        - M: object containing simulation data
        - data_path: path to the data directory

    Returns:
        None
    """
    area_list = M.simulation.params["areas_simulated"]
    label = M.simulation.label
    
    for area in area_list:
        compute_rate_time_series(M, data_path, label, area, 'full')
    
    # Time series of firing rates
    rate_time_series = {}
    for area in area_list:
        fn = os.path.join(data_path, label,
                          'Analysis',
                          'rate_time_series_full',
                          'rate_time_series_full_{}.npy'.format(area))
        rate_time_series[area] = np.load(fn)
        

    t_min = 500.
    t_max = M.simulation.params['t_sim']
    matrix = []
    for area in area_list:
        binned_spikes = rate_time_series[area][500:600]
        matrix.append(binned_spikes)
    
    matrix = np.array(matrix)
    
    fig = pl.figure(figsize=(12, 5))
    fig.suptitle('Instantaneous firing rate over simulated areas', fontsize=16, x=0.45, y=0.95)
    ax = pl.subplot()
    
    cmap = pl.get_cmap('YlOrBr')

    ax.patch.set_hatch('x')
    im = ax.pcolormesh(matrix, cmap=cmap, edgecolors='None', vmin=0)
    ax.set_xlim(0, matrix[0].size)
    
    ax.set_xticks([i for i in np.arange(0, 100, 10)])
    ax.set_xticklabels([i for i in np.arange(500, 600, 10)])
    ax.set_yticks([a + 0.5 for a in list(range(len(area_list)))])
    ax.set_yticklabels(area_list)
    ax.set_ylabel('Area', size=13)
    ax.set_xlabel('Time (ms)', size=13)

    cbar = pl.colorbar(im)
    cbar.set_label('spikes/s', fontsize=13)