import json
import numpy as np
import os
import correlation_toolbox.helper as ch

def compute_corrcoeff(M, data_path, label):
    """
    Compute the correlation coefficient between 
    the spiking rates of different populations 
    in different areas.
    
    Parameters:
        - M (MultiAreaModel): The MultiAreaModel instance.
        - data_path (str): The path to the directory where the data is stored.
        - label (str): The label used to identify the specific data set.
    
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

    tmin = 500.
    subsample = 2000
    resolution = 1.

    cc_dict = {}
    for area in areas_simulated:
        cc_dict[area] = {}
        for pop in M.structure[area]:
            fp = '-'.join((label,
                           'spikes',  # assumes that the default label for spike files was used
                           area,
                           pop))
            fn = '{}/{}.npy'.format(load_path, fp)
            # +1000 to ensure that we really have subsample non-silent neurons in the end
            spikes = np.load(fn, allow_pickle=True)
            ids = np.unique(spikes[:, 0])
            dat = ch.sort_gdf_by_id(spikes, idmin=ids[0], idmax=ids[0]+subsample+1000)
            bins, hist = ch.instantaneous_spike_count(dat[1], resolution, tmin=tmin, tmax=T)
            rates = ch.strip_binned_spiketrains(hist)[:subsample]
            
            # Test if only 1 of the neurons is firing, if yes, print warning message and continue
            if rates.shape[0] < 2:
                print(f"WARNING: There are less than 2 neurons firing in the population: {area} {pop}, the corresponding cross-correlation will not be computed.")
                continue
            
            # Compute cross correlation coefficient
            cc = np.corrcoef(rates)
            cc = np.extract(1-np.eye(cc[0].size), cc)
            cc[np.where(np.isnan(cc))] = 0.
            cc_dict[area][pop] = np.mean(cc)

    fn = os.path.join(save_path,
                      'corrcoeff.json')
    with open(fn, 'w') as f:
        json.dump(cc_dict, f)
