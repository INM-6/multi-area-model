import numpy as np
import matplotlib.pyplot as pl

def plot_instan_mean_firing_rate(M):
    # load spike data and calculate instantaneous and mean firing rates
    data = np.loadtxt(M.simulation.data_dir + '/recordings/' + M.simulation.label + "-spikes-1-0.dat", skiprows=3)
    tsteps, spikecount = np.unique(data[:,1], return_counts=True)
    rate = spikecount / M.simulation.params['dt'] * 1e3 / np.sum(M.N_vec)
    
    # visualize calculate instantaneous and mean firing rates
    width = 10
    panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

    height = width / panel_wh_ratio * float(nrows) / ncols
    pl.rcParams['figure.figsize'] = (width, height)

    fig = pl.figure()
    fig, ax = pl.subplots()
    ax.plot(tsteps, rate)
    ax.plot(tsteps, np.average(rate)*np.ones(len(tsteps)), label='mean')
    ax.set_title('Instantaneous and mean firing rate across all populations')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('firing rate (spikes / s)')
    ax.set_xlim(0, M.simulation.params['t_sim'])
    ax.set_ylim(0, 50)
    ax.legend()