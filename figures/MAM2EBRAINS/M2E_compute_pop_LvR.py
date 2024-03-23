import json
import numpy as np
import os

from multiarea_model.analysis_helpers import pop_LvR

def compute_pop_LvR(M, data_path, label):
    """
    Compute LvR for the entire network from raw spike
    files of a given simulation.

    Parameters:
        - M (MultiAreaModel): MultiAreaModel instance
        - data_path (str): path to the data
        - label (str): label for the data

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


    pop_LvR_dict = {}

    for area in areas_simulated:
        pop_LvR_dict[area] = {}
        LvR_list = []
        N = []
        for pop in M.structure[area]:
            fp = '-'.join((label,
                           'spikes',  # Assumes that the default label for spike files was used
                           area,
                           pop))
            fn = '{}/{}.npy'.format(load_path, fp)
            dat = np.load(fn, allow_pickle=True)
            pop_LvR_dict[area][pop] = pop_LvR(dat, 2., 500., T, round(M.N[area][pop]))[0]

    fn = os.path.join(save_path,
                      'pop_LvR.json')
    with open(fn, 'w') as f:
        json.dump(pop_LvR_dict, f)
