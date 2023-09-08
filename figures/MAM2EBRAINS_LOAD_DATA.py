def load_data(M, A):
    # load spike data and calculate instantaneous and mean firing rates
    data = np.loadtxt(M.simulation.data_dir + '/recordings/' + M.simulation.label + "-spikes-1-0.dat", skiprows=3)
    tsteps, spikecount = np.unique(data[:,1], return_counts=True)
    firing_rate = spikecount / M.simulation.params['dt'] * 1e3 / np.sum(M.N_vec)

    
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
    

    """
    Calculate time-averaged population rates and store them in member pop_rates.
    If the rates had previously been stored with the same
    parameters, they are loaded from file.

    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    compute_stat : bool, optional
        If set to true, the mean and variance of the population rate
        is calculated. Defaults to False.
        Caution: Setting to True slows down the computation.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    """
    A.create_pop_rates()
    print("Computing population rates done")

    
    """
    Calculate synchrony as the coefficient of variation of the population rate
    and store in member synchrony. Uses helper function synchrony.
    If the synchrony has previously been stored with the
    same parameters, they are loaded from file.


    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    resolution : float, optional
        Resolution of the population rate. Defaults to 1 ms.
    """
    A.create_synchrony()
    print("Computing synchrony done")

    
    """
    Calculate poulation-averaged LvR (see Shinomoto et al. 2009) and
    store as member pop_LvR. Uses helper function LvR.

    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    """
    A.create_pop_LvR()
    print("Computing population LvR done")
    
    
    """
    Calculate time series of population- and area-averaged firing rates.
    Uses ah.pop_rate_time_series.
    If the rates have previously been stored with the
    same parameters, they are loaded from file.


    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
    """
    A.create_rate_time_series()
    print("Computing rate time series done")

    A.save()
    
    return tsteps, firing_rate