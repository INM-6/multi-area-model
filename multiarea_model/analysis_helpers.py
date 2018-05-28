# -*- coding: utf-8 -*-

"""
analysis_helpers
============

Helper and analysis functions to support ana_vistools and
the analysis of simulations of the multi-area model of
macaque visual cortex (Schmidt et al. 2018).


Functions
--------
_create_parameter_dict : Create parameter dict for functions
                         of data class.
_check_stored_data : Check if stored data was computed
                     with the correct Parameters.
online_hist : Compute spike histogram on a spike file line by line.
pop_rate : Compute average firing rate.
pop_rate_distribution : Compute distribution of single-cell firing rates.
pop_rate_time_series : Compute time series of population rate.
Regularity measures:
    - pop_cv_isi : Compute population-averaged CV ISI.
    - pop_LvR: Compute average LvR of neuronal population.

Synchrony measures :
    - synchrony : CV of population rate.
    - synchrony_subthreshold : Synchrony measure on membrane potentials.
    - spike_synchrony : Synchrony measure on population rate.

spectrum : Compound power spectrum of a neuronal population.
synaptic_output : Synaptic output of neuronal population.
compare : Compare two simulations with each other.


Authors
--------
Maximilian Schmidt
Sacha van Albada

"""

from copy import copy
from nested_dict import nested_dict
import numpy as np
import json
from itertools import product
from scipy.signal import welch


area_list = ['V1', 'V2', 'VP', 'V3', 'PIP', 'V3A', 'MT', 'V4t', 'V4',
             'PO', 'VOT', 'DP', 'MIP', 'MDP', 'MSTd', 'VIP', 'LIP',
             'PITv', 'PITd', 'AITv', 'MSTl', 'FST', 'CITv', 'CITd',
             '7a', 'STPp', 'STPa', 'FEF', '46', 'TF', 'TH', 'AITd']
pop_list = ['23E', '23I', '4E',  '4I', '5E', '5I', '6E', '6I']


def model_iter(mode='single',
               areas=None, pops='complete',
               areas2=None, pops2='complete'):
    """
    Helper function to create a an iterator over all possible pairs of
    populations in the model, possible restricted by specifying areas
    or pops.

    Parameters
    ----------
    mode : {'single', 'pairs'}, optional
        If equal to 'single', loop over all populations of all areas.
        If equal to 'pairs', loop over all pairs of
        populations of all areas.
        Defaults to 'single'.

    areas, areas2 : list, optional
        If specified, loop only over these areas as target and source
        areas. Defaults to None, which corresponds to taking all areas
        into account.
    pops, pops2 : string or list, optional
        If specified, loop only over these populations as target and
        source populations. Defaults to 'complete', which corresponds
        to taking all areas into account. If None, loop only over
        areas.

    Returns
    -------
    iterator : iterator
        Cartesian product of 2 ('single' mode) or 4 ('double' mode) lists
    """
    if mode == 'single':
        assert((areas2 is None) and (pops2 is 'complete'))
    if pops is None or pops2 is None:
        assert((pops is None) and (pops2 is None) or mode == 'single')
    if pops == 'complete':
        pops = pop_list
    if areas is None:
        areas = area_list
    if pops2 == 'complete':
        pops2 = pop_list
    if areas2 is None:
        areas2 = area_list
    if mode == 'single':
        if pops is None:
            return product(areas)
        else:
            return product(areas, pops)
    elif mode == 'pairs':
        if pops is None:
            return product(areas, areas2)
        else:
            return product(areas, pops, areas2, pops2)


def area_spike_train(spike_data):
    """
    Helper function to create one spike train for an area from
    the spike trains of the single populations.

    Parameters
    ----------
    spike_data : dict
        Dictionary containing the populations as keys
        and their spike trains as values. Spike trains
        are stored as 2D arrays with GIDs in the 1st column
        and time stamps in the 2nd column.

    Returns
    -------
    data_array : numpy.ndarray
    """
    data_array = np.array([])
    for pop in spike_data:
        data_array = np.append(data_array, spike_data[pop])
    data_array = np.reshape(data_array, (-1, 2))
    return data_array


def centralize(data, time=False, units=False):
    """
    Code written by David Dahmen and Jakob Jordan,
    available from https://github.com/INM-6/correlation-toolbox .

    Set mean of the given data to zero by averaging either
    across time or units.
    """

    assert(time is not False or units is not False)
    res = copy(data)
    if time is True:
        res = np.array([x - np.mean(x) for x in res])
    if units is True:
        res = np.array(res - np.mean(res, axis=0))
    return res


def sort_gdf_by_id(data, idmin=None, idmax=None):
    """
    Code written by David Dahmen and Jakob Jordan,
    available from https://github.com/INM-6/correlation-toolbox .

    Sort gdf data [(id,time),...] by neuron id.

    Parameters
    ----------

    data: numpy.array (dtype=object) with lists ['int', 'float']
          The nest output loaded from gdf format. Each row contains a
          global id
    idmin, idmax : int, optional
            The minimum/maximum neuron id to be considered.

    Returns
    -------
    ids : list of ints
          Neuron ids, e.g., [id1,id2,...]
    srt : list of lists of floats
          Spike trains corresponding to the neuron ids, e.g.,
          [[t1,t2,...],...]
    """

    assert((idmin is None and idmax is None)
           or (idmin is not None and idmax is not None))

    if len(data) > 0:
        # get neuron ids
        if idmin is None and idmax is None:
            ids = np.unique(data[:, 0])
        else:
            ids = np.arange(idmin, idmax+1)
        srt = []
        for i in ids:
            srt.append(np.sort(data[np.where(data[:, 0] == i)[0], 1]))
        return ids, srt
    else:
        print('CT warning(sort_spiketrains_by_id): empty gdf data!')
        return None, None


"""
Helper functions for data loading
"""


def _create_parameter_dict(default_dict, T, **keywords):
    """
    Create the parameter dict for the members of the data class.

    Parameters
    ----------
    default_dict : dict
        Default dictionary of the function calling this function.
    T : float
        Maximal time of the simulation of the data class calling.

    Returns
    -------
    d : dict
        Parameter dictionary.
    """
    d = default_dict
    if 't_min' not in keywords:
        t_min = 500.
        d.update({'t_min': t_min})
    if 't_max' not in keywords:
        t_max = T
        d.update({'t_max': t_max})
    d.update(keywords)
    return d


def _check_stored_data(fp, fn_iter, param_dict):
    """
    Check if a data member of the data class has already
    been computed with the same parameters.

    Parameters
    ----------
    fn : string
        Filename of the file containing the data.
    param_dict : dict
        Parameters of the calculation to compare with
        the parameters of the stored data.
    """
    if 'json' in fp:
        try:
            f = open(fp)
            data = json.load(f)
            f.close()
        except IOError:
            return None
        param_dict2 = data['Parameters']
    else:
        try:
            data = _load_npy_to_dict(fp, fn_iter)
        except IOError:
            return None
        with open('-'.join((fp, 'parameters')), 'r') as f:
            param_dict2 = json.load(f)
    param_dict_copy = copy(param_dict)
    param_dict2_copy = copy(param_dict2)
    for k in param_dict:
        if (isinstance(param_dict_copy[k], list) or
                isinstance(param_dict_copy[k], np.ndarray)):
            param_dict_copy[k] = set(param_dict_copy[k])
        if (isinstance(param_dict2_copy[k], list) or
                isinstance(param_dict2_copy[k], np.ndarray)):
            param_dict2_copy[k] = set(param_dict2_copy[k])
    if param_dict_copy == param_dict2_copy:
        print("Loading data from file")
        return data
    else:
        print("Stored data have been computed "
              "with different parameters")
        return None


def _save_dict_to_npy(fp, data):
    """
    Save data dictionary to binary numpy files
    by iteratively going through the dictionary.

    Parameters
    ----------
    fp : str
       File pattern to which the keys of the dictionary are attached.
    data : dict
       Dictionary containing the data
    """
    for key, val in data.items():
        if key != 'Parameters':
            fp_key = '-'.join((fp, key))
            if isinstance(val, dict):
                _save_dict_to_npy(fp_key, val)
            else:
                np.save(fp_key, val)
        else:
            fp_key = '-'.join((fp, 'parameters'))
            with open(fp_key, 'w') as f:
                json.dump(val, f)


def _load_npy_to_dict(fp, fn_iter):
    """
    Load data stored in the files defined by fp
    and fn_iter to a dictionary.

    Parameters
    ----------
    fp : str
       Base file pattern of the npy files
    fn_iter : iterable
       Iterable defining all the suffixes that are
       appended to fp to form the file names.
    """
    data = nested_dict()
    for it in fn_iter:
        fp_it = (fp,) + it
        fp_ = '{}.npy'.format('-'.join(fp_it))
        if len(it) == 1:
            data[it[0]] = np.load(fp_)
        else:
            data[it[0]][it[1]] = np.load(fp_)
    return data


"""
Analysis functions
"""


def pop_rate(data_array, t_min, t_max, num_neur, return_stat=False):
    """
    Calculates firing rate of a given array of spikes.
    Rates are calculated in spikes/s. Assumes spikes are sorted
    according to time. First calculates rates for individual neurons
    and then averages over neurons.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time stamp to be considered in ms.
    tmax : float
        Maximal time stamp to be considered in ms.
    num_neur : int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    Returns
    -------
    mean : float
        Mean firing rate across neurons.
    std : float
        Standard deviation of firing rate distribution.
    rates : list
        List of single-cell firing rates.
    """

    indices = np.where(np.logical_and(data_array[:, 1] > t_min,
                                      data_array[:, 1] < t_max))
    data_array = data_array[indices]
    if return_stat:
        rates = []
        for i in np.unique(data_array[:, 0]):
            num_spikes = np.where(data_array[:, 0] == i)[0].size
            rates.append(num_spikes / ((t_max - t_min) / 1000.))
            while len(rates) < num_neur:
                rates.append(0.0)
            mean = np.mean(rates)
            std = np.std(rates)
            return mean, std, rates
    else:
        return data_array[:, 1].size / (num_neur * (t_max - t_min) / 1000.)


def pop_rate_distribution(data_array, t_min, t_max, num_neur):
    """
    Calculates firing rate distribution over neurons in a given array
    of spikes. Rates are calculated in spikes/s. Assumes spikes are
    sorted according to time. First calculates rates for individual
    neurons and then averages over neurons.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time stamp to be considered in ms.
    tmax : float
        Maximal time stamp to be considered in ms.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.

    Returns
    -------
    bins : numpy.ndarray
        Left edges of the distribution bins
    vals : numpy.ndarray
        Values of the distribution
    mean : float
        Arithmetic mean of the distribution
    std : float
        Standard deviation of the distribution
    """
    indices = np.where(np.logical_and(data_array[:, 1] > t_min,
                                      data_array[:, 1] < t_max))
    neurons = data_array[:, 0][indices]
    neurons = np.sort(neurons)
    if len(neurons) > 0:
        n = neurons[0]
    else:  # No spikes in [t_min, t_max]
        n = None
    rates = np.zeros(int(num_neur))
    s = 0
    for i in range(neurons.size):
        if neurons[i] == n:
            rates[s] += 1
        else:
            n = neurons[i]
            s += 1
    rates /= (t_max - t_min) / 1000.
    vals, bins = np.histogram(rates, bins=100)
    vals = vals / float(np.sum(vals))
    if (num_neur > 0. and t_max != t_min
            and len(data_array) > 0 and len(indices) > 0):
        return bins[0:-1], vals, np.mean(rates), np.std(rates)
    else:
        return np.arange(0, 20., 20. / 100.), np.zeros(100), 0.0, 0.0


def pop_rate_time_series(data_array, num_neur, t_min, t_max,
                         resolution=10., kernel='binned'):
    """
    Computes time series of the population-averaged rates of a group
    of neurons.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time for the calculation.
    tmax : float
        Maximal time for the calculation.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    kernel : {'gauss_time_window', 'alpha_time_window',
        'rect_time_window'}, optional
        Specifies the kernel to be
        convolved with the spike histogram. Defaults to 'binned',
        which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
        Defaults to 1 ms.

    Returns
    -------
    time_series : numpy.ndarray
        Time series of the population rate
    """
    if kernel == 'binned':
        rate, times = np.histogram(data_array[:, 1], bins=int((t_max - t_min) / (resolution)),
                                   range=(t_min + resolution / 2., t_max + resolution / 2.))
        rate = rate / (num_neur * resolution / 1000.0)
        rates = np.array([])
        last_time_step = times[0]

        for i in range(1, times.size):
            rates = np.append(
                rates, rate[i - 1] * np.ones_like(np.arange(last_time_step, times[i], 1.0)))
            last_time_step = times[i]

        time_series = rates
    else:
        spikes = data_array[:, 1][data_array[:, 1] > t_min]
        spikes = spikes[spikes < t_max]
        binned_spikes = np.histogram(spikes, bins=int(
            (t_max - t_min)), range=(t_min, t_max))[0]
        if kernel == 'rect_time_window':
            kernel = np.ones(int(resolution)) / resolution
        if kernel == 'gauss_time_window':
            sigma = resolution
            time_range = np.arange(-0.5 * (t_max - t_min),
                                   0.5 * (t_max - t_min), 1.0)
            kernel = 1 / (np.sqrt(2.0 * np.pi) * sigma) * \
                np.exp(-(time_range ** 2 / (2 * sigma ** 2)))
        if kernel == 'alpha_time_window':
            alpha = 1 / resolution
            time_range = np.arange(-0.5 * (t_max - t_min),
                                   0.5 * (t_max - t_min), 1.0)
            time_range[time_range < 0] = 0.0
            kernel = alpha * time_range * np.exp(-alpha * time_range)

        rate = np.convolve(kernel, binned_spikes, mode='same')
        rate = rate / (num_neur / 1000.0)
        time_series = rate

    return time_series


def pop_cv_isi(data_array, t_min, t_max):
    """
    Calculate coefficient of variation of interspike intervals
    between t_min and t_max for every single neuron in data_array
    and average the result over neurons in data_array.
    Assumes spikes are sorted according to time.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time stamp to be considered in ms.
    tmax : float
        Maximal time stamp to be considered in ms.

    Returns
    -------
    mean : float
        Mean CV ISI value of the population
    """
    cv_isi = []
    indices = np.where(np.logical_and(data_array[:, 1] > t_min,
                                      data_array[:, 1] < t_max))[0]
    if len(data_array) > 1 and len(indices) > 1:
        for i in np.unique(data_array[:, 0]):
            intervals = np.diff(data_array[indices][
                                np.where(data_array[indices, 0] == i), 1])
            if (len(intervals) > 0):
                cv_isi.append(np.std(intervals) / np.mean(intervals))
        if len(cv_isi) > 0:
            return np.mean(cv_isi)
        else:
            return 0.0
    else:
        print('cv_isi: no or only one spike in data_array, returning 0.0')
        return 0.0


def ISI_SCC(data_array, t_min, t_max):
    """
    Computes the serial correlation coefficient of
    inter-spike intervals of the given spike data.

    Parameters
    ----------
    data_array : numpy.ndarray
        Arrays with spike data.
        column 0: neuron_ids, column 1: spike times
    t_min : float
        Minimal time for the calculation.
    t_max : float
        Maximal time for the calculation.


    Return
    -------
    bins : numpy.ndarray
        ISI lags
    values : numpy.ndarray
        Serial correlation coefficient values
    """
    indices = np.where(np.logical_and(data_array[:, 1] > t_min,
                                      data_array[:, 1] < t_max))
    scc_averaged = np.zeros(max(1001, 2 * (t_max - t_min) + 1))
    half = max(1000, 2 * (t_max - t_min)) / 2.0
    if len(data_array) > 1 and len(indices) > 1:
        for i in np.unique(data_array[:, 0]):
            intervals = np.diff(data_array[indices][
                                np.where(data_array[indices, 0] == i), 1])

            if intervals.size > 1:
                mean = np.mean(intervals)
                scc = (np.correlate(intervals, intervals, mode='full') - mean ** 2) / (
                    np.mean(intervals ** 2) - mean ** 2)
                scc_averaged[half - scc.size /
                             2:half + scc.size / 2 + 1] += scc

        scc_averaged = scc_averaged / np.unique(data_array[:, 0]).size
        return np.arange(-half, half + 1, 1), scc_averaged / np.sum(scc_averaged)
    else:
        print('cv_isi: no or only one spike in data_array, returning 0.0')
        return 0.0


def pop_LvR(data_array, t_ref, t_min, t_max, num_neur):
    """
    Compute the LvR value of the given data_array.
    See Shinomoto et al. 2009 for details.

    Parameters
    ----------
    data_array : numpy.ndarray
        Arrays with spike data.
        column 0: neuron_ids, column 1: spike times
    t_ref : float
        Refractory period of the neurons.
    t_min : float
        Minimal time for the calculation.
    t_max : float
        Maximal time for the calculation.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.

    Returns
    -------
    mean : float
        Population-averaged LvR.
    LvR : numpy.ndarray
        Single-cell LvR values
    """
    i_min = np.searchsorted(data_array[:, 1], t_min)
    i_max = np.searchsorted(data_array[:, 1], t_max)
    LvR = np.array([])
    data_array = data_array[i_min:i_max]
    for i in np.unique(data_array[:, 0]):
        intervals = np.diff(data_array[
                            np.where(data_array[:, 0] == i)[0], 1])
        if intervals.size > 1:
            val = np.sum((1. - 4 * intervals[0:-1] * intervals[1:] / (intervals[0:-1] + intervals[
                         1:]) ** 2) * (1 + 4 * t_ref / (intervals[0:-1] + intervals[1:])))
            LvR = np.append(LvR, val * 3 / (intervals.size - 1.))
        else:
            LvR = np.append(LvR, 0.0)
    if len(LvR) < num_neur:
        LvR = np.append(LvR, np.zeros(num_neur - len(LvR)))
    return np.mean(LvR), LvR


def synchrony(data_array, num_neur, t_min, t_max, resolution=1.0):
    """
    Compute the synchrony of an array of spikes as the coefficient
    of variation of the population rate.
    Uses pop_rate_time_series().


    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time for the calculation of the histogram in ms.
    tmax : float
        Maximal time for the calculation of the histogram in ms.
    resolution : float, optional
        Bin width of the histogram. Defaults to 1 ms.

    Returns
    -------
    synchrony : float
        Synchrony of the population.
    """
    spike_count_histogramm = pop_rate_time_series(
        data_array, num_neur, t_min, t_max, resolution=resolution)
    mean = np.mean(spike_count_histogramm)
    variance = np.var(spike_count_histogramm)
    synchrony = variance / mean
    return synchrony


def spectrum(data_array, num_neur, t_min, t_max, resolution=1., kernel='binned', Df=None):
    """
    Compute compound power spectrum of a population of neurons.
    Uses the powerspec function of the correlation toolbox.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    t_min : float
        Minimal time for the calculation of the histogram in ms.
    t_max : float
        Maximal time for the calculation of the histogram in ms.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
        Defaults to 1 ms.
    Df : float, optional
        Window width of sliding rectangular filter (smoothing) of the spectrum.
        The default value is None and leads to no smoothing.

    Returns
    -------
    power : numpy.ndarray
        Values of the power spectrum.
    freq : numpy.ndarray
        Discrete frequency values
    """
    rate = pop_rate_time_series(
        data_array, num_neur, t_min, t_max, kernel=kernel, resolution=resolution)
    rate = centralize(rate, units=True)
    freq, power = welch(rate, fs=1.e3,
                        noverlap=1000, nperseg=1024)
    return power[0][freq > 0], freq[freq > 0]


def synaptic_output(rate, tau_syn, t_min, t_max, resolution=1.):
    """
    Compute the synaptic output of a population of neurons.
    Convolves the population spike histogram with an exponential
    synaptic filter.

    Parameters
    ----------
    rate : numpy.ndarray
        Time series of the population rate.
    tau_syn : float
        Synaptic time constant of the single neurons
    t_min : float
        Minimal time for the calculation.
    t_max : float
        Maximal time for the calculation.
    resolution : float, optional
        Time resolution of the synaptic filtering kernel in ms.
        Defaults to 1 ms.

    """
    t = np.arange(0., 20., resolution)
    kernel = np.exp(-t / tau_syn)
    syn_current = np.convolve(kernel, rate, mode='same')
    return syn_current


def compare(sim1, sim2):
    """
    Compares two simulations (2 instances of the ana_vistools data class)
    in regards of their parameters.

    Parameters
    ----------
    sim1, sim2 : ana_vistools.data
        The two instances of the ana_vistools.data classe to be compared.

    Returns
    -------
    None
    """

    template = "{0:30}{1:20}{2:25}{3:15}"
    print(template.format("parameter", sim1.label[0:5], sim2.label[0:5], "equal?"))

    info1 = sim1.sim_info
    info2 = sim2.sim_info
    compare_keys = []
    for key in list(info1.keys()) + list(info2.keys()):
        p = False
        if key in list(info1.keys()):
            value = info1[key]
        else:
            value = info2[key]

        if isinstance(value, str):
            # To exclude paths from the compared keys
            if (value.find('_') == -1 and
                value.find('/') == -1 and
                    value.find('.') == -1):
                p = True
        else:
            p = True

        if key in ['sim_label', 'K_stable_path']:
            p = False
        if p and key not in compare_keys:
            compare_keys.append(key)
    for key in compare_keys:
        if key in info2 and key in info1:
            out = (key, str(info1[key]),  str(info2[
                    key]), info1[key] == info2[key])
        elif key not in info1:
            out = (key, '', str(info2[key]), 'false')
        elif key not in info2:
            out = (key, str(info1[key]),  '', 'false')
        print(template.format(*out))
    # Compare sum of indegrees
    s1 = 0.
    s2 = 0.
    for area1 in sim1.areas:
        for pop1 in sim1.structure[area1]:
            for area2 in sim1.areas:
                for pop2 in sim1.structure[area2]:
                    s1 += sim1.indegree_data[area1][pop1][area2][pop2]
                    s2 += sim2.indegree_data[area1][pop1][area2][pop2]

    out = ('indegrees', str(s1),  str(s2), s1 == s2)
    print(template.format(*out))
