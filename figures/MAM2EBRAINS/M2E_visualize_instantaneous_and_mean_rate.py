def plot_instan_mean_firing_rate(M):
    # load spike data and calculate instantaneous and mean firing rates
    data = np.loadtxt(M.simulation.data_dir + '/recordings/' + M.simulation.label + "-spikes-1-0.dat", skiprows=3)
    tsteps, spikecount = np.unique(data[:,1], return_counts=True)
    firing_rate = spikecount / M.simulation.params['dt'] * 1e3 / np.sum(M.N_vec)
    
    # visualize calculate instantaneous and mean firing rates
    ax = pl.subplot()
    ax.plot(tsteps, rate)
    ax.plot(tsteps, np.average(rate)*np.ones(len(tsteps)), label='mean')
    ax.set_title('Instantaneous and mean firing rate across all populations')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('firing rate (spikes / s)')
    ax.set_xlim(0, M.simulation['t_sim'])
    ax.set_ylim(0, 50)
    ax.legend()