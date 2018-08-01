import json
import numpy as np
import os
import pyx

from helpers import original_data_path
from multiarea_model import MultiAreaModel
from matrix_plot import matrix_plot, rate_histogram_plot

import pylab as pl
from matplotlib import gridspec
from matplotlib import rc_file
rc_file('plotstyle.rc')

"""
Figure layout
"""
nrows = 2
ncols = 4
width = 7.0866
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio
height = 6.
pl.rcParams['figure.figsize'] = (width, height)


fig = pl.figure()
gs1 = gridspec.GridSpec(6, 1)
gs1.update(left=0.08, right=0.55, top=0.95,
           bottom=0.5, wspace=0., hspace=0.2)
ax_rates = []
ax_rates.append(pl.subplot(gs1[:1, 0:1]))
ax_rates.append(pl.subplot(gs1[1:2, 0:1]))
ax_rates.append(pl.subplot(gs1[2:3, 0:1]))
ax_rates.append(pl.subplot(gs1[3:4, 0:1]))
ax_rates.append(pl.subplot(gs1[4:5, 0:1]))
ax_rates.append(pl.subplot(gs1[5:6, 0:1]))

gs2 = gridspec.GridSpec(1, 1)
gs2.update(left=0.83, right=0.95, top=0.95,
           bottom=0.5, wspace=0., hspace=0.2)
ax_EV = pl.subplot(gs2[:, :])

gs4 = gridspec.GridSpec(1, 1)
gs4.update(left=0.08, right=0.7, top=0.4,
           bottom=0.04, wspace=0., hspace=0.2)
ax_sketch = pl.subplot(gs4[:, :])
ax_sketch.spines['right'].set_color('none')
ax_sketch.spines['top'].set_color('none')
ax_sketch.spines['left'].set_color('none')
ax_sketch.spines['bottom'].set_color('none')
ax_sketch.yaxis.set_ticks_position("none")
ax_sketch.xaxis.set_ticks_position("none")
ax_sketch.set_xticks([])
ax_sketch.set_yticks([])

gs3 = gridspec.GridSpec(6, 1)
gs3.update(left=0.62, right=0.75, top=0.95,
           bottom=0.5, wspace=0., hspace=0.2)
ax_phasespace = []
ax_phasespace.append(pl.subplot(gs3[:1, 0:1]))
ax_phasespace.append(pl.subplot(gs3[1:2, 0:1]))
ax_phasespace.append(pl.subplot(gs3[2:3, 0:1]))
ax_phasespace.append(pl.subplot(gs3[3:4, 0:1]))
ax_phasespace.append(pl.subplot(gs3[4:5, 0:1]))
ax_phasespace.append(pl.subplot(gs3[5:6, 0:1]))

gs4 = gridspec.GridSpec(2, 1)
gs4.update(left=0.72, right=0.96, top=0.4,
           bottom=0.04, wspace=0., hspace=0.25)
ax_matrix = pl.subplot(gs4[:1, :])
ax_hist = pl.subplot(gs4[1:2, :])
ax_hist.spines['right'].set_color('none')
ax_hist.spines['top'].set_color('none')
ax_hist.spines['left'].set_color('none')
ax_hist.spines['bottom'].set_color('none')
ax_hist.yaxis.set_ticks_position("none")
ax_hist.xaxis.set_ticks_position("none")
ax_hist.set_xticks([])
ax_hist.set_yticks([])

for ax, label in zip([ax_rates[0], ax_phasespace[0], ax_EV, ax_sketch, ax_matrix],
                     ['A', 'B', 'C', 'D', 'E']):
    if label == 'C':
        label_pos = [-0.1, 1.]
    else:
        label_pos = [-0.1, 1.01]

    ax.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
            fontdict={'fontsize': 10.,
                      'weight': 'bold',
                      'horizontalalignment': 'left',
                      'verticalalignment': 'bottom'},
            transform=ax.transAxes)

"""
Load data
"""
chi_list = [1.0, 1.8, 1.9, 2., 2.1, 2.5]

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

LOAD_ORIGINAL_DATA = True

if LOAD_ORIGINAL_DATA:
    labels = ['33fb5955558ba8bb15a3fdce49dfd914682ef3ea',
              '1474e1884422b5b2096d3b7a20fd4bdf388af7e0',
              '99c0024eacc275d13f719afd59357f7d12f02b77',
              'f18158895a5d682db5002489d12d27d7a974146f',
              '08a3a1a88c19193b0af9d9d8f7a52344d1b17498',
              '5bdd72887b191ec22a5abcc04ca4a488ea216e32']

    label_stat_rate = '99c0024eacc275d13f719afd59357f7d12f02b77'
    data_path = original_data_path
else:
    from network_simulations import init_models
    from config import data_path
    models = init_models('Fig4')
    labels = [M.simulation.label for M in models]
    label_stat_rate = labels[2]  # chi=1.9

rate_time_series = {label: {} for label in labels}
rate_time_series_pops = {label: {} for label in labels}
for label in labels:
    for area in M.area_list:
        fn = os.path.join(data_path, label,
                          'Analysis',
                          'rate_time_series_full',
                          'rate_time_series_full_{}.npy'.format(area))
        rate_time_series[label][area] = np.load(fn)
        rate_time_series_pops[label][area] = {}
        for pop in M.structure[area]:
            fn = os.path.join(data_path, label,
                              'Analysis',
                              'rate_time_series_full',
                              'rate_time_series_full_{}_{}.npy'.format(area, pop))
            rate_time_series_pops[label][area][pop] = np.load(fn)

    with open(os.path.join(data_path, label,
                           'Analysis',
                           'rate_time_series_full',
                           'rate_time_series_full_Parameters.json')) as f:
        rate_time_series[label]['Parameters'] = json.load(f)

# stationary firing rates
fn = os.path.join(data_path, label_stat_rate, 'Analysis', 'pop_rates.json')
with open(fn, 'r') as f:
    pop_rates = {label_stat_rate: json.load(f)}


# Meanfield part: first initialize base class to compute initial rates
# and then compute analytical rates for all configurations of chi
K_stable_path = '../SchueckerSchmidt2017/K_prime_original.npy'

conn_params = {'g': -12.,
               'cc_weights_factor': 1.,
               'cc_weights_I_factor': 1.,
               'K_stable': K_stable_path,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.125 * 10 / 3. - 7 / 3.,
               'fac_nu_ext_TH': 1.2}
input_params = {'rate_ext': 10.}
network_params = {'connection_params': conn_params,
                  'input_params': input_params}

initial_rates = np.zeros(254)
theory_params = {'T': 30.,
                 'dt': 0.01,
                 'initial_rates': initial_rates}

M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
p, r_base = M.theory.integrate_siegert()

"""
Plotting
"""
print("Plotting rate time series")
area = 'V1'
for i, (cc, label) in enumerate(zip(chi_list, labels)):
    ax = ax_rates[i]
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.plot(rate_time_series[label][area], lw=1, color='k')
    ax.set_xlim((0., 5e4))
    ax.set_ylim((-5., 60.))
    ax.set_yticks([5., 40.])
    ax.text(51500., 48., r'$\chi = $' + str(cc))
    if i == len(labels) - 1:
        ax.vlines(0., 0., 40.)
        ax.hlines(0., 0., 2.)

        ax.set_xlabel('Time (s)')
        ax.set_xticks([0., 1e4, 2e4, 3e4, 4e4, 5e4])
        ax.set_xticklabels([0, 10, 20, 30, 40, 50])
    else:
        ax.set_xticks([])
    if i == 3:
        ax.set_ylabel(r'Rate $(\mathrm{spikes/s})$')

print("Plotting critical eigenvalues")
lambda_max = []
analytical_rates = {}
for chi, label in zip(chi_list[:-1], labels[:-1]):
    if chi == 1.:
        chi_I = 1.
    else:
        chi_I = 2.

    conn_params = {'g': -12.,
                   'cc_weights_factor': chi,
                   'cc_weights_I_factor': chi_I,
                   'K_stable': K_stable_path,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.125 * 10 / 3. - 7 / 3.,
                   'fac_nu_ext_TH': 1.2}
    input_params = {'rate_ext': 10.}
    network_params = {'connection_params': conn_params,
                      'input_params': input_params}

    initial_rates = np.zeros(254)
    theory_params = {'T': 30.,
                     'dt': 0.01,
                     'initial_rates': initial_rates}

    M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)

    pops, rates_full = M.theory.integrate_siegert()
    analytical_rates[chi] = rates_full[:, -1]
    ana_rates = analytical_rates[chi]
    lambda_max.append(M.theory.lambda_max(ana_rates))

ax = ax_EV
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

ax.plot(lambda_max, np.arange(len(lambda_max)), '.', ms=5)
ax.vlines(1., 0, 4)
ax.set_ylim((-0.5, 4.5))

ax.invert_yaxis()
ax.set_xlabel(r'$\mathrm{max}\{\mathrm{Re}\left(\lambda_i\right)\}$')
ax.set_ylabel(r'$\chi$')
ax.set_xticks([0.5, 1.])
ax.set_yticks(np.arange(0., 5.))
ax.set_yticklabels(chi_list)

        
load_path = 'Fig4_theory_data'
for i, cc_weights_factor in enumerate(chi_list):
    ax = ax_phasespace[i]
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([0., 5., 35.])
    if i != len(labels) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'Average activity ($\mathrm{spikes/s}$)')
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])

    # Trim data to 10000 samples to have the same sample number for all configurations
    data = np.load(os.path.join(load_path,
                                'results_{}.npy'.format(cc_weights_factor)))
    vals, bins = np.histogram(
        np.mean(data[:, :, -1], axis=1), bins=50, range=(0, 40))

    ax.bar(bins[:-1], vals, width=np.diff(bins)
           [0], color='k', edgecolor='none')

print("Plotting rate matrix")
label = '99c0024eacc275d13f719afd59357f7d12f02b77'

matrix = np.zeros((len(M.area_list), 8))
for i, area in enumerate(M.area_list):
    for j, pop in enumerate(M.structure['V1'][::-1]):
        if pop not in M.structure[area]:
            rate = np.nan
        else:
            rate = pop_rates[label][area][pop][0]
        if rate == 0.0:
            rate = 1e-5
        matrix[i][j] = rate
matrix = np.transpose(matrix)
matrix_plot(fig, ax_matrix, matrix, position='single')

pos = ax_hist.get_position()
ax_hist_pos = [pos.x0, pos.y0, pos.x1 - pos.x0, pos.y1 - pos.y0]
rate_histogram_plot(fig, ax_hist_pos, matrix, position='single')

pl.savefig('Fig4_metastability_mpl.eps')

"""
Merge with sketch figure
"""
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(
    0.5, 0.5, "Fig4_metastability_mpl.eps", width=17.6))
c.insert(pyx.epsfile.epsfile(
    0.8, 1., "Fig4_metastability_phasespace_sketch.eps", width=12.2))

c.writeEPSfile("Fig4_metastability.eps")
