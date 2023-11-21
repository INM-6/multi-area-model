import numpy as np
import matplotlib.pyplot as pl

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

    ax.set_title('Instantaneous and mean firing rate across all populations', fontsize=15, pad=10)
    ax.set_xlabel('Time (ms)', fontsize=13)
    ax.set_ylabel('Firing rate (spikes / s)', fontsize=12)
    ax.set_xlim(0, M.simulation.params['t_sim'])
    ax.set_ylim(0, 50)
    ax.legend()
    pl.show()