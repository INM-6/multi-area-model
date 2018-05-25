import correlation_toolbox.helper as ch
import json
import numpy as np
import os
import sys

from multiarea_model.multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask
from scipy.stats import levene
from statsmodels.tsa.vector_ar.var_model import VAR


"""
Compute the conditional Granger causality to a given population of an
area based on the population-averaged spike rates from a given
simulation.
"""

data_path = sys.argv[1]
label = sys.argv[2]
area = sys.argv[3]
pop = sys.argv[4]
target_pair = (area, pop)

load_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'rate_time_series_full')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'granger_causality')
try:
    os.mkdir(save_path)
except FileExistsError:
    pass

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    sim_params = json.load(f)
T = sim_params['T']

"""
Create MultiAreaModel instance to have access to data structures
"""
connection_params = {'g': -11.,
                     'cc_weights_factor': sim_params['cc_weights_factor'],
                     'cc_weights_I_factor': sim_params['cc_weights_I_factor'],
                     'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'connection_params': connection_params}
M = MultiAreaModel(network_params)
# We exclude external input from the analysis
K = M.K_matrix[:, :-1]


def indices_to_population(structure, indices):
    complete = []
    for area in M.area_list:
        for pop in structure[area]:
            complete.append(area + '-' + pop)

    complete = np.array(complete)
    return complete[indices]


if pop not in M.structure[area]:
    gc = {}
else:
    rate_time_series = {}
    for source_area in M.area_list:
        rate_time_series[source_area] = {}
        for source_pop in M.structure[source_area]:
            fn = os.path.join(load_path,
                              'rate_time_series_full_{}_{}.npy'.format(source_area, source_pop))
            dat = np.load(fn)
            rate_time_series[source_area][source_pop] = dat
    fn = os.path.join(load_path,
                      'rate_time_series_full_Parameters.json')
    with open(fn, 'r') as f:
        rate_time_series['Parameters'] = json.load(f)

    tmin, tmax = (500., T)
    imax = int(tmax - rate_time_series['Parameters']['t_min'])
    imin = int(tmin - rate_time_series['Parameters']['t_min'])


    # Order of vector auto-regressive model

    # As potentially Granger-causal populations, we only consider source
    # population with an indegree > 1
    mask = create_mask(M.structure, target_pops=[pop], target_areas=[area], external=False)[:, :-1]
    pairs = indices_to_population(M.structure, np.where(K[mask] > 1.))

    # Build a list of the time series of all source pairs onto the target pair
    all_rates = [ch.centralize(rate_time_series[area][pop][imin:imax], units=True)]
    target_index = 0
    source_pairs = [target_pair]
    for pair in pairs:
        source_area = pair.split('-')[0]
        source_pop = pair.split('-')[1]
        if (source_area, source_pop) != target_pair:
            all_rates.append(ch.centralize(rate_time_series[source_area][source_pop][imin:imax],
                                           units=True))
            source_pairs.append((source_area, source_pop))

    # Fit VAR with all rates
    dat = np.vstack(all_rates)
    dat = dat.transpose()
    model = VAR(dat)
    # Order of auto-regressive regression model
    selected_order = 25
    res = model.fit(selected_order)
    Sigma_matrix = np.cov(res.resid.transpose())
    # Residual variance of the target population in the VAR incl. all time
    # series
    variance = Sigma_matrix[target_index][target_index]

    dim = res.resid[:, 0].size
    k = dat.shape[1] * selected_order

    # Now we loop through all source pairs, compute the reduced VAR
    # (neglecting the time series of that source pair) and then compute
    # the conditional Granger causality based on this result
    # causality, significance, res = [], [], []
    gc = {area: {} for area in M.area_list}
    for source_index, source_pair in enumerate(source_pairs):
        if source_pair != target_pair:
            print(source_pair)
            source_area = source_pair[0]
            source_pop = source_pair[1]
            # Fit marginal VAR
            dat_reduced = np.vstack(all_rates[:source_index] + all_rates[source_index+1:])
            source_pairs_reduced = source_pairs[:source_index] + source_pairs[source_index+1:]
            dat_reduced = dat_reduced.transpose()
            model_reduced = VAR(dat_reduced)
            res_reduced = model_reduced.fit(selected_order)

            Sigma_matrix_reduced = np.cov(res_reduced.resid.transpose())
            target_index_reduced = source_pairs_reduced.index(target_pair)
            # Compute the conditional Granger causality as the log-ratio of the residual variances
            variance_reduced = Sigma_matrix_reduced[target_index_reduced][target_index_reduced]
            cause = np.log(variance_reduced / variance)

            k_reduced = dat_reduced.shape[1] * selected_order

            # Test if the residual variances are significantly different
            test = levene(np.sqrt((dim - 1.)/(dim - k)) * res.resid[:, target_index],
                          np.sqrt((dim - 1.)/(dim - k_reduced)) * res_reduced.resid[:, target_index_reduced])
        else:
            cause = np.nan
            test = (np.nan, np.nan)
            res_red = np.nan

            gc[source_area][source_pop] = (cause, test[1])

fn = os.path.join(save_path,
                  'granger_causality_{}_{}.json'.format(area, pop))
with open(fn, 'w') as f:
    json.dump(gc, f)
