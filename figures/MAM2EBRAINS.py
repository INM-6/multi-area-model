# Instantaneous and mean firing rate across all populations

def plot_instan_mean_firing _rate(tsteps, rate):
    fig, ax = plt.subplots()
    ax.plot(tsteps, rate)
    ax.plot(tsteps, np.average(rate)*np.ones(len(tsteps)), label='mean')
    ax.set_title('Instantaneous and mean firing rate across all populations')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('firing rate (spikes / s)')
    ax.set_xlim(0, sim_params['t_sim'])
    ax.set_ylim(0, 50)
    ax.legend()
    
def plot_raster_plot(A):
    """
    Create raster display of a single area with populations stacked onto each other. Excitatory neurons in blue, inhibitory neurons in red.

    Parameters
    ----------
    area : string {area}
        Area to be plotted.
    frac_neurons : float, [0,1]
        Fraction of cells to be considered.
    t_min : float, optional
        Minimal time in ms of spikes to be shown. Defaults to 0 ms.
    t_max : float, optional
        Minimal time in ms of spikes to be shown. Defaults to simulation time.
    output : {'pdf', 'png', 'eps'}, optional
        If given, the function stores the plot to a file of the given format.
    """
    t_min = 0.
    t_max = 500.
    areas = ['V1', 'V2', 'FEF']
    frac_neurons = 1.
    for area in areas:
        A.single_dot_display(area,  frac_neurons, t_min, t_max)