import json
import numpy as np
import os

from multiarea_model.analysis_helpers import pop_rate

def compute_pop_rates(M, data_path, label):
    """
    Compute stationary spike rates for the entire network from raw spike
    files of a given simulation.

    Parameters:
        - M (MultiAreaModel): The MultiAreaModel instance containing the simulation data.
        - data_path (str): The path to the directory where the simulation data is stored.
        - label (str): The label used to identify the simulation data.

    Returns:
        None
    """
    load_path = os.path.join(data_path,
                             label,
                             'recordings')
    save_path = os.path.join(data_path,
                             label,
                             'Analysis')

    T = M.simulation.params["t_sim"]
    areas_simulated = M.simulation.params["areas_simulated"]

    pop_rates = {}
    for area in areas_simulated:
        pop_rates[area] = {}
        rate_list = []
        N = []
        for pop in M.structure[area]:
            fp = '-'.join((label,
                           'spikes',  # assumes that the default label for spike files was used
                           area,
                           pop))
            fn = '{}/{}.npy'.format(load_path, fp)
            dat = np.load(fn, allow_pickle=True)
            pop_rates[area][pop] = pop_rate(dat, 500., T, M.N[area][pop])
            rate_list.append(pop_rates[area][pop])
            N.append(M.N[area][pop])
        pop_rates[area]['total'] = np.average(rate_list, weights=N)

    fn = os.path.join(save_path,
                      'pop_rates.json')
    with open(fn, 'w') as f:
        json.dump(pop_rates, f)
