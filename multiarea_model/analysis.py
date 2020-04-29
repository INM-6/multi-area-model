# -*- coding: utf-8 -*-

"""
analysis
============

Analysis package to load and analyse data from simulations of the
multi-area model of macaque visual cortex (Schmidt et al. 2017).


Classes
--------

Analysis : loads the data of the specified simulation and provides members
functions to post-process the data and plot it in various visualizations.

Authors
--------
Maximilian Schmidt
Sacha van Albada

"""
from . import analysis_helpers as ah
import glob
import inspect
from itertools import chain, product
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from copy import copy
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
from nested_dict import nested_dict

try:
    import seaborn as sns
except ImportError:
    pass


class Analysis:
    def __init__(self, network, simulation, data_list=['spikes'],
                 load_areas=None):
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

        self.network = network
        self.simulation = simulation
        assert(self.network.label == self.simulation.network.label)
        self.output_dir = os.path.join(self.simulation.data_dir, 'Analysis/')
        try:
            os.mkdir(self.output_dir)
        except OSError:
            pass

        self.T = self.simulation.T

        self.areas_simulated = self.simulation.areas_simulated
        self.areas_recorded = self.simulation.areas_recorded
        if load_areas is None:
            self.areas_loaded = self.areas_simulated
        else:
            self.areas_loaded = load_areas

        assert(all([area in self.areas_recorded for area in
                    self.areas_loaded])), "Tried to load areas which "
        "were not recorded"
        self.interarea_speed = self.network.params['delay_params']['interarea_speed']

        self.load_data(data_list)

    def load_data(self, data_list):
        """
        Loads simulation data of the requested type either from hdf5 files.

        Parameters
        ----------

        data_list : list
            list of observables to be loaded. Can contain 'spikes' and 'vm'
        """
        rec_dir = os.path.join(self.simulation.data_dir, 'recordings')
        self.network_gids = pd.read_csv(os.path.join(rec_dir, 'network_gids.txt'),
                                        names=['area', 'population', 'min_gid', 'max_gid'])
        for data_type in data_list:
            if data_type == 'spikes':
                columns = ['senders', 'times']
                d = 'spike_dict'
            elif data_type == 'vm':
                assert(self.simulation.params['recording_dict']['record_vm']), "Trying to "
                "load membrane potentials, but these data have not been recorded"
                d = 'vm_dict'
                columns = ['senders', 'times', 'V_m']
            print('loading {}'.format(data_type))
            data = {}
            # Check if the data has already been stored in binary file
            for area in self.areas_loaded:
                data[area] = {}
                for pop in self.network.structure[area]:
                    fp = '-'.join((self.simulation.label,
                                   self.simulation.params['recording_dict'][d]['label'],
                                   area,
                                   pop))
                    fn = os.path.join(rec_dir,
                                      '.'.join((fp, 'npy')))
                    try:
                        data[area][pop] = np.load(fn, allow_pickle=True)
                    except FileNotFoundError:
                        if not hasattr(self, 'all_spikes'):
                            fp = '.'.join(('-'.join((self.simulation.label,
                                                     self.simulation.params[
                                                         'recording_dict'][d]['label'],
                                                     '*')),
                                           'gdf'))
                            files = glob.glob(os.path.join(rec_dir, fp))
                            dat = pd.DataFrame(columns=columns)
                            for f in files:
                                dat = dat.append(pd.read_csv(f,
                                                             names=columns, sep='\t',
                                                             index_col=False),
                                                 ignore_index=True)
                            self.all_spikes = dat
                        print(area, pop)
                        gids = self.network_gids[(self.network_gids.area == area) &
                                                 (self.network_gids.population == pop)]
                        ind = ((self.all_spikes.senders >= gids.min_gid.values[0]) &
                               (self.all_spikes.senders <= gids.max_gid.values[0]))
                        dat = self.all_spikes[ind]
                        self.all_spikes.drop(np.where(ind)[0])
                        np.save(fn, np.array(dat))
                        data[area][pop] = np.array(dat)
            if data_type == 'spikes':
                self.spike_data = data
            elif data_type == 'vm':
                # Sort membrane potentials to reduce data load
                self.vm_data = {}
                for area in data:
                    self.vm_data[area] = {}
                    for pop in data[area]:
                        neurons, time, vm = ah.sort_membrane_by_id(data[area][pop])
                        self.vm_data[area][pop] = {'neurons': neurons,
                                                   'V_m': vm,
                                                   'time': (time[0], time[-1])}
                self._set_num_vm_neurons()

    def _set_num_vm_neurons(self):
        """
        Sets number of neurons from which membrane voltages
        were recorded during simulation.
        """
        self.num_vm_neurons = {}
        for area in self.vm_data:
            self.num_vm_neurons[area] = {}
            for pop in self.vm_data[area]:
                self.num_vm_neurons[area][pop] = self.vm_data[area][pop][
                    'neurons'][-1] - self.vm_data[area][pop]['neurons'][0] + 1

# ______________________________________________________________________________
# Functions for post-processing data into dynamical measures
    def create_pop_rates(self, **keywords):
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
        default_dict = {'areas': self.areas_loaded,
                        'pops': 'complete', 'compute_stat': False}
        params = ah._create_parameter_dict(default_dict, self.T, **keywords)
        iterator = ah.model_iter(mode='single',
                                 areas=params['areas'],
                                 pops=params['pops'])
        # Check if population rates have been stored with the same parameters
        fp = os.path.join(self.output_dir, 'pop_rates.json')
        self.pop_rates = ah._check_stored_data(fp,
                                               copy(iterator), params)

        if self.pop_rates is None:
            print("Computing population rates")
            d = nested_dict()
            d['Parameters'] = params

            if params['compute_stat']:
                for area in params['areas']:
                    if params['pops'] == 'complete':
                        pops = self.network.structure[area]
                    else:
                        pops = params['pops']
                        total_rates = []
                        for pop in pops:
                            rate = ah.pop_rate(self.spike_data[area][pop],
                                               params['t_min'],
                                               params['t_max'],
                                               self.network.N[area][pop])
                            d[area][pop] = (rate[0], rate[1])
                            total_rates += rate[2]
                        d[area]['total'] = (np.mean(total_rates), np.std(total_rates))
            else:
                for area, pop in iterator:
                    if pop in self.network.structure[area]:
                        spikes = self.spike_data[area][pop][:, 1]
                        indices = np.where(np.logical_and(spikes > params['t_min'],
                                                          spikes < params['t_max']))
                        d[area][pop] = (indices[0].size / (self.network.N[
                            area][pop] * (params['t_max'] - params['t_min']) / 1000.0), np.nan)
                    else:
                        d[area][pop] = (0., 0.)
                for area in params['areas']:
                    total_spikes = ah.area_spike_train(self.spike_data[area])
                    indices = np.where(np.logical_and(total_spikes[:, 1] > params['t_min'],
                                                      total_spikes[:, 1] < params['t_max']))
                    d[area]['total'] = total_spikes[:, 1][indices].size / (
                        self.network.N[area]['total'] *
                        (params['t_max'] - params['t_min']) / 1000.0)
            self.pop_rates = d.to_dict()

    def create_pop_rate_dists(self, **keywords):
        """
        Calculate single neuron population rates and store them in member pop_rate_dists.
        If the distributions had previously been stored with the
        same parameters, they are loaded from file.
        Uses helper function pop_rate_distribution.

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

        default_dict = {'areas': self.areas_loaded, 'pops': 'complete'}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)
        iterator = ah.model_iter(mode='single',
                                 areas=params['areas'],
                                 pops=params['pops'])
        elements = [('histogram',), ('stats-mu',), ('stats-sigma',)]
        iter_list = [tuple(chain.from_iterable(prod)) for
                     prod in product(copy(iterator), elements)]
        # Check if population rates have been stored with the same parameters
        self.pop_rate_dists = ah._check_stored_data(os.path.join(self.output_dir,
                                                                 'pop_rate_dists'),
                                                    iter_list, params)

        if self.pop_rate_dists is None:
            print("Computing population dists")
            d = nested_dict()
            d['Parameters'] = params
            for area, pop in iterator:
                if pop in self.network.structure[area]:
                    res = list(ah.pop_rate_distribution(self.spike_data[area][pop],
                                                        params['t_min'],
                                                        params['t_max'],
                                                        self.network.N[area][pop]))
                    d[area][pop] = {'histogram': np.array([res[0], res[1]]),
                                    'stats': {'mu': res[2],
                                              'sigma': res[3]}}
            self.pop_rate_dists = d.to_dict()

    def create_synchrony(self, **keywords):
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

        default_dict = {'areas': self.areas_loaded,
                        'pops': 'complete', 'resolution': 1.0}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)
        iterator = ah.model_iter(mode='single',
                                 areas=params['areas'],
                                 pops=params['pops'])
        # Check if synchrony values have been stored with the same parameters
        self.synchrony = ah._check_stored_data(os.path.join(self.output_dir, 'synchrony.json'),
                                               copy(iterator), params)

        if self.synchrony is None:
            print("Computing synchrony")
            d = nested_dict()
            d['Parameters'] = params
            for area, pop in iterator:
                if pop in self.network.structure[area]:
                    d[area][pop] = ah.synchrony(self.spike_data[area][pop],
                                                self.network.N[area][pop],
                                                params['t_min'],
                                                params['t_max'],
                                                resolution=params['resolution'])
                else:
                    d[area][pop] = np.nan

            for area in params['areas']:
                total_spikes = ah.area_spike_train(self.spike_data[area])
                d[area]['total'] = ah.synchrony(
                    total_spikes,
                    self.network.N[area]['total'],
                    params['t_min'],
                    params['t_max'],
                    resolution=params['resolution'])
            self.synchrony = d.to_dict()

    def create_rate_time_series(self, **keywords):
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
        default_dict = {'areas': self.areas_loaded, 'pops': 'complete',
                        'kernel': 'binned', 'resolution': 1.0}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)

        # Check if firing rates have been stored with the same parameters
        fp = os.path.join(self.output_dir, 'rate_time_series')
        iterator_areas = ah.model_iter(mode='single',
                                       areas=params['areas'],
                                       pops=None)
        iterator_pops = ah.model_iter(mode='single',
                                      areas=params['areas'],
                                      pops=params['pops'])
        self.rate_time_series = ah._check_stored_data(fp, copy(iterator_areas), params)
        fp = os.path.join(self.output_dir, 'rate_time_series_pops')
        self.rate_time_series_pops = ah._check_stored_data(fp, copy(iterator_pops), params)

        if self.rate_time_series is None:
            print('Computing rate time series')

            # calculate area-averaged firing rates
            d = nested_dict()
            d['Parameters'] = params
            # population-averaged firing rates
            d_pops = nested_dict()
            d_pops['Parameters'] = params
            for area, pop in iterator_pops:
                if pop in self.network.structure[area]:
                    time_series = ah.pop_rate_time_series(self.spike_data[area][pop],
                                                          self.network.N[area][pop],
                                                          params['t_min'],
                                                          params['t_max'],
                                                          params['resolution'],
                                                          kernel=params['kernel'])
                else:
                    time_series = np.nan*np.ones(params['t_max'] - params['t_min'])
                d_pops[area][pop] = time_series

                total_spikes = ah.area_spike_train(self.spike_data[area])
                time_series = ah.pop_rate_time_series(total_spikes,
                                                      self.network.N[area]['total'],
                                                      params['t_min'],
                                                      params['t_max'],
                                                      params['resolution'],
                                                      kernel=params['kernel'])
                d[area] = time_series
            self.rate_time_series_pops = d_pops.to_dict()
            self.rate_time_series = d.to_dict()

    def create_synaptic_input(self, **keywords):
        """
        Calculate synaptic input of populations and areas using the spike data.
        Uses function ah.pop_synaptic_input.
        If the synaptic inputs have previously been stored with the
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
            Convolution kernel for the calculation of the underlying firing rates.
            Defaults to 'binned' which corresponds to a simple histogram.
        resolution: float, optional
            Width of the convolution kernel. Specifically it correponds to:
            - 'binned' : bin width of the histogram
            - 'gauss_time_window' : sigma
            - 'alpha_time_window' : time constant of the alpha function
            - 'rect_time_window' : width of the moving rectangular function
        """
        default_dict = {'areas': self.areas_loaded, 'pops': 'complete',
                        'resolution': 1., 'kernel': 'binned'}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)

        # Check if synaptic inputs have been stored with the same parameters
        iterator_areas = ah.model_iter(mode='single',
                                       areas=params['areas'],
                                       pops=None)
        iterator_pops = ah.model_iter(mode='single',
                                      areas=params['areas'],
                                      pops=params['pops'])
        fp = os.path.join(self.output_dir, 'synaptic_input')
        self.synaptic_input = ah._check_stored_data(fp, copy(iterator_areas), params)
        fp = os.path.join(self.output_dir, 'synaptic_input_pops')
        self.synaptic_input_pops = ah._check_stored_data(fp, copy(iterator_pops), params)

        if self.synaptic_input is None:
            print('Computing rate time series')
            if 'rate_time_series' not in inspect.getmembers(self):
                self.create_rate_time_series(**params)

            d_pops = nested_dict()
            d_pops['Parameters'] = params
            for area, pop in copy(iterator_pops):
                if pop in self.network.structure[area]:
                    if 'I' in pop:
                        tau_syn = self.network.params['neuron_params'][
                            'single_neuron_dict']['tau_syn_in']
                    else:
                        tau_syn = self.network.params['neuron_params'][
                            'single_neuron_dict']['tau_syn_ex']
                    time_series = ah.synaptic_output(self.rate_time_series_pops[area][pop],
                                                     tau_syn, params['t_min'], params['t_max'],
                                                     resolution=params['resolution'])
                    d_pops[area][pop] = time_series
            self.synaptic_output_pops = d_pops.to_dict()

            d_pops = nested_dict()
            d_pops['Parameters'] = params
            d_pops['Parameters'] = params
            for area, pop in iterator_pops:
                if pop in self.network.structure[area]:
                    time_series = np.zeros(
                        int((params['t_max'] - params['t_min']) / params['resolution']))
                    for source_area, source_pop in ah.model_iter(mode='single',
                                                                 areas=self.areas_loaded):
                        if source_pop in self.network.structure[source_area]:
                            weight = self.network.W[area][pop][source_area][source_pop]
                            time_series += (self.synaptic_output_pops[source_area][source_pop] *
                                            abs(weight) *
                                            self.network.K[area][pop][source_area][source_pop])
                    d_pops[area][pop] = time_series

            d = nested_dict()
            d['Parameters'] = params
            for area in params['areas']:
                d[area] = np.zeros(
                    int((params['t_max'] - params['t_min']) / params['resolution']))
                for pop in self.network.structure[area]:
                    d[area] += d_pops[area][pop] * self.network.N[area][pop]
                d[area] /= self.network.N[area]['total']
            self.synaptic_input = d.to_dict()
            self.synaptic_input_pops = d_pops.to_dict()

    def create_pop_cv_isi(self, **keywords):
        """
        Calculate population-averaged CV ISI values and store as member pop_cv_isi.
        Uses helper function cv_isi.
        If the CV ISI have previously been stored with the
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
        """

        default_dict = {'areas': self.areas_loaded, 'pops': 'complete'}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)
        # Check if CV ISI have been stored with the same parameters
        iterator = ah.model_iter(mode='single',
                                 areas=params['areas'],
                                 pops=params['pops'])
        fp = os.path.join(self.output_dir, 'pop_cv_isi.json')
        self.pop_cv_isi = ah._check_stored_data(fp,
                                                copy(iterator), params)

        if self.pop_cv_isi is None:
            print("Computing population CV ISI")
            d = nested_dict()
            d['Parameters'] = params
            for area, pop in iterator:
                if pop in self.network.structure[area]:
                    d[area][pop] = ah.pop_cv_isi(self.spike_data[area][pop],
                                                 params['t_min'],
                                                 params['t_max'])
            self.pop_cv_isi = d.to_dict()

    def create_pop_LvR(self, **keywords):
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
        default_dict = {'areas': self.areas_loaded, 'pops': 'complete'}
        params = ah._create_parameter_dict(
            default_dict, self.T, **keywords)

        # Check if LvR have been stored with the same parameters
        iterator = ah.model_iter(mode='single',
                                 areas=params['areas'],
                                 pops=params['pops'])
        fp = os.path.join(self.output_dir, 'pop_LvR.json')
        self.pop_LvR = ah._check_stored_data(fp,
                                             copy(iterator), params)
        if self.pop_LvR is None:
            print("Computing population LvR")
            d = nested_dict()
            d['Parameters'] = params
            for area, pop in iterator:
                if pop in self.network.structure[area]:
                    if self.network.N[area][pop] > 0.:
                        d[area][pop] = ah.pop_LvR(self.spike_data[area][pop],
                                                  2.0,
                                                  params['t_min'],
                                                  params['t_max'],
                                                  int(self.network.N[area][pop]))[0]
            self.pop_LvR = d.to_dict()

# ______________________________________________________________________________
# Function for plotting data
    def single_dot_display(self, area,  frac_neurons, t_min=500., t_max='T', **keywords):
        """
        Create raster display of a single area with populations stacked
        onto each other. Excitatory neurons in blue, inhibitory
        neurons in red.

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
        if t_max == 'T':
            t_max = self.T

        try:
            fig = plt.figure()
        except RuntimeError:
            plt.switch_backend('agg')
            fig = plt.figure()
        ax = fig.add_subplot(111)
        assert(area in self.areas_loaded)
        # Determine number of neurons that will be plotted for this area (for vertical offset)
        offset = 0
        n_to_plot = {}
        for pop in self.network.structure[area]:
            n_to_plot[pop] = int(self.network.N[
                                 area][pop] * frac_neurons)
            offset = offset + n_to_plot[pop]
        y_max = offset + 1
        prev_pop = ''
        yticks = []
        yticklocs = []
        # Loop over populations
        for pop in self.network.structure[area]:
            if pop[0:-1] != prev_pop:
                prev_pop = pop[0:-1]
                yticks.append('L' + pop[0:-1])
                yticklocs.append(offset - 0.5 * n_to_plot[pop])
            indices = np.where(np.logical_and(self.spike_data[area][pop] > t_min,
                                              self.spike_data[area][pop] < t_max))

            pop_data = self.spike_data[area][pop][indices[0]]
            neurons_to_plot = np.arange(np.min(self.spike_data[area][pop][:, 0]), np.min(
                self.spike_data[area][pop][:, 0]) + n_to_plot[pop], 1)
            # print pop,neurons_to_plot.size

            if pop.find('E') > (-1):
                pcolor = '#595289'
            else:
                pcolor = '#af143c'

            for k in range(n_to_plot[pop]):
                spike_times = pop_data[
                    pop_data[:, 0] == neurons_to_plot[k], 1]

                ax.plot(spike_times, np.zeros(len(spike_times)) +
                        offset - k, '.', color=pcolor, markersize=1)
            offset = offset - n_to_plot[pop]
        y_min = offset
        ax.set_xlim([t_min, t_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel('time [ms]', size=16)
        ax.set_ylabel('Neuron', size=16)

        if 'output' in keywords:
            plt.savefig(os.path.join(self.output_dir,
                                     '{}_Dotplot_{}.{}'.format(self.simulation.label,
                                                               area, keywords['output'])))
        else:
            fig.show()

    def single_rate_display(self, area, pop=None,  t_min=None, t_max=None, **keywords):
        """
        Plot rates time series for a single area or population.
        Uses rate time series stored in dictionary pop_rate_time_series.
        Parameters
        ----------
        area : string {area}
            Area to be plotted.
        pop : string, optional
            If given, the rate of a specific population in area is plotted.
            Defaults to None, then the area-averaged rate is plotted.
        t_min : float, optional
            Minimal time in ms of spikes to be shown.
            Defaults to minimal time of computed rate time series.
        t_max : float, optional
            Minimal time in ms of spikes to be shown.
            Defaults to maximal time of computed rate time series.
        output : {'pdf', 'png', 'eps'}, optional
            If given, the function stores the plot to a file of the given format.
        """
        if pop is None:
            rates = self.rate_time_series[area]
            params = self.rate_time_series['Parameters']
        else:
            rates = self.rate_time_series_pops[area][pop]
            params = self.rate_time_series_pops['Parameters']

        if t_max is None:
            t_max = params['t_max']
        if t_min is None:
            t_min = params['t_min']

        i_min = int(t_min - params['t_min'])
        i_max = int(t_max - params['t_min'])

        rates = rates[i_min:i_max]

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        times = np.arange(t_min, t_max, 1.0)

        ax.plot(times, rates, color='k', markersize=1)

        if pop:
            ax.set_title('{} {} {}'.format(area, pop, params['kernel']))
        else:
            ax.set_title('{} {}'.format(area, params['kernel']))
        ax.set_xlabel('time [ms]', size=15)
        ax.set_ylabel('rate [1/s]', size=15)

        if 'output' in keywords:
            if pop:
                plt.savefig(os.path.join(self.output_dir,
                                         '{}_rate_{}_{}.{}'.format(self.simulation.label,
                                                                   area, pop, keywords['output'])))
            else:
                plt.savefig(os.path.join(self.output_dir,
                                         '{}_rate_{}.{}'.format(self.simulation.label,
                                                                area, keywords['output'])))
        else:
            fig.show()

    def single_power_display(self, area, pop=None, t_min=None,
                             t_max=None, resolution=1., kernel='binned', Df=None, **keywords):
        """
        Plot power spectrum for a single area.
        Directly computes the values via function 'spectrum' using
        rate time series stored in dictionary pop_rate_time_series.

        Parameters
        ----------
        area : string {area}
            Area to be plotted.
        pop : string, optional
            If given, the rate of a specific population in area is plotted.
            Defaults to None, then the area-averaged rate is plotted.
        t_min : float, optional
            Minimal time in ms of spikes to be shown.
            Defaults to minimal time of underlying rate time series.
        t_max : float, optional
            Minimal time in ms of spikes to be shown.
            Defaults to maximal time of underlying rate time series.
        kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
            Specifies the kernel to be convolved with the spike histogram.
            Defaults to 'binned', which corresponds to no convolution.
        resolution: float, optional
            Width of the convolution kernel. Specifically it correponds to:
            - 'binned' : bin width of the histogram
            - 'gauss_time_window' : sigma
            - 'alpha_time_window' : time constant of the alpha function
            - 'rect_time_window' : width of the moving rectangular function
        Df : float, optional
            Window width of sliding rectangular filter (smoothing) of the spectrum.
            The default value is None and leads to no smoothing.
        output : {'pdf', 'png', 'eps'}, optional
            If given, the function stores the plot to a file of the given format.
        """
        if pop is None:
            data = self.spike_data[area][self.network.structure[area][0]]
            num_neur = self.network.N[area][self.network.structure[area][0]]
            for population in self.network.structure[area][1:]:
                data = np.vstack((data, self.spike_data[area][population]))
                num_neur += self.network.N[area][self.network.structure[area][0]]
        else:
            data = self.spike_data[area][pop]
            num_neur = self.network.N[area][pop]

        if t_max is None:
            t_max = self.T
        if t_min is None:
            t_min = 0.

        power, freq = ah.spectrum(data, num_neur, t_min, t_max,
                                  resolution=resolution, kernel=kernel, Df=Df)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(freq, power, color='k', markersize=3)
        if pop:
            ax.set_title('{} {} {}'.format(area, pop, kernel))
        else:
            ax.set_title('{} {}'.format(area, kernel))
        ax.set_xlabel('Frequency [Hz]', size=16)
        ax.set_ylabel('Power', size=16)
        ax.set_xlim(0.0, 500.0)
        ax.set_yscale("Log")

        if 'output' in keywords:
            if pop:
                plt.savefig(os.path.join(self.output_dir,
                                         '{}_power_spectrum_{}_{}.{}'.format(self.simulation.label,
                                                                             area,
                                                                             pop,
                                                                             keywords['output'])))
            else:
                plt.savefig(os.path.join(self.output_dir,
                                         '{}_power_spectrum_{}.{}'.format(self.simulation.label,
                                                                          area,
                                                                          keywords['output'])))
        else:
            fig.show()

    def show_rates(self, area_list=None, **keywords):
        """
        Plot overview over time-averaged population rates encoded in colors
        with areas along x-axis and populations along y-axis.

        Parameters
        ----------
        area_list : list, optional
           Specifies with areas are plotted in which order.
           Default to None, leading to plotting of  all areas ordered by architectural type.
        output : {'pdf', 'png', 'eps'}, optional
            If given, the function stores the plot to a file of the given format.
        """
        if area_list is None:
            area_list = ['V1', 'V2', 'VP', 'V3', 'PIP', 'V3A', 'MT', 'V4t', 'V4',
                                     'PO', 'VOT', 'DP', 'MIP', 'MDP', 'MSTd', 'VIP', 'LIP',
                                     'PITv', 'PITd', 'AITv', 'MSTl', 'FST', 'CITv', 'CITd',
                                     '7a', 'STPp', 'STPa', 'FEF', '46', 'TF', 'TH', 'AITd']

        matrix = np.zeros((len(area_list), len(self.network.structure['V1'])))

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        for i, area in enumerate(area_list):
            print(i, area)
            for j, pop in enumerate(self.network.structure_reversed['V1']):
                if pop in self.network.structure[area]:
                    rate = self.pop_rates[area][pop][0]
                    if rate == 0.0:
                        rate = 1e-5  # To distinguish zero-rate from non-existing populations
                else:
                    rate = np.nan
                matrix[i][j] = rate

        cm = plt.cm.jet
        cm = cm.from_list('mycmap', [(0., 64./255., 192./255.),  # custom dark blue
                                     (0., 128./255., 192./255.),  # custom light blue
                                     'white',
                                     (245./255., 157./255., 115./255.),  # custom light red
                                     (192./255., 64./255., 0.)], N=256)  # custom dark red
        cm.set_under('0.3')
        cm.set_bad('k')

        matrix = np.transpose(matrix)
        masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
        ax.patch.set_hatch('x')
        im = ax.pcolormesh(masked_matrix, cmap=cm, edgecolors='None', norm=LogNorm(
            vmin=0.01, vmax=100.))
        ax.set_xlim(0, matrix[0].size)

        x_index = np.arange(4.5, 31.6, 5.0)
        x_ticks = [int(a + 0.5) for a in x_index]
        y_index = list(range(len(self.network.structure['V1'])))
        y_index = [a + 0.5 for a in y_index]
        print(self.network.structure['V1'])
        ax.set_xticks(x_index)
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(y_index)
        ax.set_yticklabels(self.network.structure_reversed['V1'])
        ax.set_ylabel('Population', size=18)
        ax.set_xlabel('Area index', size=18)
        t = FixedLocator([0.01, 0.1, 1., 10., 100.])

        plt.colorbar(im, ticks=t)

        if 'output' in keywords:
            plt.savefig(os.path.join(self.output_dir, '{}_rates.{}'.format(self.simulation.label,
                                                                           keywords['output'])))
        else:
            fig.show()

# ______________________________________________________________________________
# Functions to store data to file

    def save(self):
        """
        Saves all post-processed data to files.
        """
        members = inspect.getmembers(self)
        save_list_json = ['structure', 'pop_rates', 'synchrony',
                          'pop_cv_isi', 'pop_LvR',
                          'indegree_data', 'indegree_areas_data',
                          'outdegree_data', 'outdegree_areas_data']
        save_list_npy = ['pop_rate_dists', 'rate_time_series',
                         'rate_time_series_pops', 'bold_signal',
                         'synaptic_input', 'synaptic_input_pops']
        for i in range(0, len(members)):
            if members[i][0] in save_list_json:
                f = open(self.output_dir + members[i][0] + '.json', 'w')
                print(members[i][0])
                json.dump(members[i][1], f)
                f.close()
            if members[i][0] in save_list_npy:
                f = self.output_dir + members[i][0]
                ah._save_dict_to_npy(f, members[i][1])
