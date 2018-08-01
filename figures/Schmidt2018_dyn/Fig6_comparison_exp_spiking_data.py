import numpy as np
import os
import scipy.io as io

from helpers import original_data_path, chu2014_path
from plotcolors import myred, myblue, mypurple, mygreen, myyellow
from scipy.signal import spectrogram, welch

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib import rc_file
rc_file('plotstyle.rc')


"""
Figure layout
"""
nrows = 2
ncols = 3
width = 7.0866
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

height = width / panel_wh_ratio * float(nrows) / ncols
pl.rcParams['figure.figsize'] = (width, height)


fig = pl.figure()
axes = {}
gs1 = gridspec.GridSpec(2, 3)
gs1.update(left=0.1, right=0.95, top=0.95, wspace=0.5, bottom=0.1)
axes['A'] = pl.subplot(gs1[0, 0])
axes['B'] = pl.subplot(gs1[0, 1])
axes['C'] = pl.subplot(gs1[0, 2])
axes['D'] = pl.subplot(gs1[1, 0])
axes['E'] = pl.subplot(gs1[1, 1])
axes['F'] = pl.subplot(gs1[1, 2])


for label in list(axes.keys()):
    label_pos = [-0.25, 1.01]
    pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
            fontdict={'fontsize': 10, 'weight': 'bold',
                      'horizontalalignment': 'left', 'verticalalignment':
                      'bottom'}, transform=axes[label].transAxes)
    axes[label].spines['right'].set_color('none')
    axes[label].spines['top'].set_color('none')
    axes[label].yaxis.set_ticks_position("left")
    axes[label].xaxis.set_ticks_position("bottom")


"""
Load data
"""
chi_list = ['1.', '2.5', '1.9', '1.9_long']
chi_labels = {'1.': r'sim, $\chi=1.0$',
              '1.9_long': r'sim, $\chi=1.9$',
              '2.5': r'sim, $\chi=2.5$'}

LOAD_ORIGINAL_DATA = True

if LOAD_ORIGINAL_DATA:
    labels = ['33fb5955558ba8bb15a3fdce49dfd914682ef3ea',
              '5bdd72887b191ec22a5abcc04ca4a488ea216e32',
              '3afaec94d650c637ef8419611c3f80b3cb3ff539',
              '99c0024eacc275d13f719afd59357f7d12f02b77']
    data_path = original_data_path
else:
    from network_simulations import init_models
    from config import data_path
    models = init_models('Fig6')
    labels = [M.simulation.label for M in models]

area = 'V1'

power_spectra = {chi: {} for chi in chi_list}
for chi, label in zip(chi_list, labels):
    fp = os.path.join(data_path,
                      label,
                      'Analysis',
                      'power_spectrum_subsample')
    power_spectra[chi] = {'f': np.load(os.path.join(fp,
                                                    'power_spectrum_subsample_freq.npy')),
                          'power': np.load(os.path.join(fp,
                                                        'power_spectrum_subsample_V1.npy'))}
rate_histograms = {chi: {} for chi in chi_list}
for chi, label in zip(chi_list, labels):
    fp = os.path.join(data_path,
                      label,
                      'Analysis',
                      'rate_histogram')

    rate_histograms[chi] = {'bins': np.load(os.path.join(fp,
                                                         'rate_histogram_bins.npy')),
                            'vals': np.load(os.path.join(fp,
                                                         'rate_histogram_V1.npy'))}

"""
Load experimental data
"""
phase_labels = {'low_fluct': 'exp, low fluct.',
                'high_fluct': 'exp, high fluct.',
                'full': 'exp, full'}
exp_data = {'spectrogram': [np.load(os.path.join(chu2014_path, 'Analysis', 'spectrogram_freq.npy')),
                            np.load(os.path.join(chu2014_path, 'Analysis', 'spectrogram_time.npy')),
                            np.load(os.path.join(chu2014_path, 'Analysis', 'spectrogram_Sxx.npy'))],
            'spike_data': np.load(os.path.join(chu2014_path, 'Analysis', 'spike_data_1mm.npy')),
            'neuron_depths': np.load(os.path.join(chu2014_path, 'Analysis', 'neuron_depths.npy')),
            'power_spectra': {'f': np.load(os.path.join(chu2014_path, 'Analysis',
                                                        'power_spectrum_freq.npy')),
                              'full': np.load(os.path.join(chu2014_path, 'Analysis',
                                                           'power_spectrum_V1_full.npy')),
                              'low_fluct': np.load(os.path.join(chu2014_path, 'Analysis',
                                                                'power_spectrum_V1_low_fluct.npy')),
                              'high_fluct': np.load(os.path.join(chu2014_path, 'Analysis',
                                                                 'power_spectrum_V1_high_fluct.npy'))},
            'rate_histograms': {'bins': np.load(os.path.join(chu2014_path, 'Analysis',
                                                             'rate_histogram_bins.npy')),
                                'low_fluct': np.load(os.path.join(chu2014_path, 'Analysis',
                                                                  'rate_histogram_low.npy')),
                                'high_fluct': np.load(os.path.join(chu2014_path, 'Analysis',
                                                                   'rate_histogram_high.npy')),
                                'full': np.load(os.path.join(chu2014_path, 'Analysis',
                                                             'rate_histogram_full.npy'))}}

"""
Plotting
"""

"""
Spectrogram of experimental spiking data
"""
print("Plotting spectrogram")
f, t, Sxx = exp_data['spectrogram']
ax = axes['A']
ind = np.where(f <= 30)[0]
im = ax.pcolormesh(Sxx[ind], norm=LogNorm(
    vmin=1e-1, vmax=1.e2), cmap=pl.get_cmap('inferno'))
cb = pl.colorbar(im, ax=ax)

cb.ax.set_ylabel('Power')
cb.ax.yaxis.set_label_coords(1.4, 0.5, transform=ax.transAxes)


ax.set_yticks(np.arange(f.size)[ind][::100])
ax.set_yticklabels([r'$0$', r'$10$', r'$20$', r'$30$'])
xticks = [5., 455., 905.]
xticklocs = [np.argmin(np.abs(t - tic)) for tic in xticks]
ax.set_xticks(xticklocs)
ax.set_xticklabels([r'$5$', r'$455$', r'$905$'])
ax.set_ylabel('Frequency (Hz)', labelpad=-0.5)
ax.set_xlabel('Time (s)', labelpad=-0.25)
ax.set_xlim((0, Sxx[ind].shape[1]))
ax.set_ylim((0, Sxx[ind].shape[0]))


"""
Raster plots experimental data
"""
print("Raster plots")
tranges = [(5000., 8000.),
           (392000., 395000.)]

for (t0, t1), phase, ax in zip(tranges,
                               ['low', 'high'],
                               [axes['B'], axes['C']]):
    for i in np.argsort(exp_data['neuron_depths']):
        ind = np.where(np.logical_and(exp_data['spike_data'][i] >= t0,
                                      exp_data['spike_data'][i] < t1))
        ax.plot(exp_data['spike_data'][i][ind],
                np.ones_like(exp_data['spike_data'][i][ind]) * i,
                '.', color='k', ms=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron', labelpad=-0.3)
    if phase == 'low':
        ax.set_xticks([5000., 6500., 8000])
        ax.set_xticklabels([r'$5.$', r'$6.5$', r'$8.$'])
    elif phase == 'high':
        ax.set_xticks([392000., 393500., 395000])
        ax.set_xticklabels([r'$392.$', r'$393.5$', r'$395.$'])
    ax.set_xlim((t0, t1))
    ax.set_ylim((0., 114.))


"""
Comparison of simulated spectra
"""
print("Plotting simulated spectra")
ax = axes['D']

sim_colors = [myred, myblue, '0.5', 'k']
for i, chi in enumerate(chi_list):
    freq = power_spectra[chi]['f']
    power = power_spectra[chi]['power']
    ind = np.where(np.logical_and(freq > 0., freq <= 60.))[0]
    ax.plot(freq[ind], power[ind], color=sim_colors[i],
            label=r'sim, $\chi = $' + str(chi))

sim_colors = [myred, myblue, 'k']
ax.set_yscale('Log')
ax.set_xlim((-10., 60.))
# ax.set_ylim((1e4, 7.e6))
ax.set_xlabel('Frequency (Hz)')
ax.set_xticks([0., 20., 40.])
ax.set_ylabel('Power')


"""
Comparison with exp. spectra
"""
print("Plotting comparison with exp. spectra")
ax = axes['E']
pos = ax.get_position()
ax_inset = pl.axes([pos.x0 + 0.10, pos.y0 + 0.28, 0.1, 0.1])

colors = [mygreen, mypurple, myyellow]

ind = np.where(np.logical_and(f > 0., f <= 60.))[0]
ind_inset = np.where(np.logical_and(f > 0., f <= 5.))[0]

for i, phase in enumerate(['low_fluct', 'high_fluct', 'full']):
    f = exp_data['power_spectra']['f']
    power = exp_data['power_spectra'][phase]

    ax.plot(f[ind], power[ind],
            color=colors[i], label='exp., {}'.format(phase))

    ax_inset.plot(f[ind_inset], power[ind_inset],
                  color=colors[i], label='exp., {}'.format(phase))

f = power_spectra['1.9_long']['f']
power = power_spectra['1.9_long']['power']

ax.plot(f[ind], power[ind], color='k', label=r'sim., $\chi=1.9$')
ax_inset.plot(f[ind_inset], power[ind_inset], color='k', label=r'sim., $\chi=1.9$')

ax.set_yscale('Log')
ax.set_xlim((-10., 60.))
ax.set_xlabel('Frequency (Hz)')
ax.set_xticks([0., 20., 40.])
ax.set_ylabel('Power')

ax.add_patch(
    mpatches.Rectangle(
        (0., 1.1e-1),   # (x,y)
        5.,          # width
        6.,          # height
        transform=ax.transData,
        fill=False
    ))
ax.arrow(5.5, 2e0, 10., 0.5, transform=ax.transData,
         head_width=0.5, head_length=2., fc='k', ec='k')

ax_inset.set_yscale('Log')
ax_inset.set_xlim((-0.5, 5.))
ax_inset.set_xticks([0., 2.5, 5.])
ax_inset.spines['right'].set_color('none')
ax_inset.spines['top'].set_color('none')
ax_inset.yaxis.set_ticks_position("left")
ax_inset.xaxis.set_ticks_position("bottom")


"""
Spike rate distributions
"""
print("Plotting rate distributions")
ax = axes['F']

for i, phase in enumerate(['low_fluct', 'high_fluct', 'full']):
    bins_exp = exp_data['rate_histograms']['bins']
    vals_exp = exp_data['rate_histograms'][phase]
    ax.plot(bins_exp[:-1], vals_exp / float(np.sum(vals_exp)),
            color=colors[i], label=phase_labels[phase])

total_rates = np.zeros(0)
area = 'V1'

sim_colors = [myred, 'k', myblue]

for i, chi in enumerate(['1.', '1.9_long', '2.5']):
    bins, vals = rate_histograms[chi]['bins'], rate_histograms[chi]['vals']
    ax.plot(bins, vals / np.sum(vals),
            color=sim_colors[i], label=chi_labels[chi])

ax.legend(loc=(0.35, 0.4), fontsize=8.)

ax.set_xlabel('Rate (spikes/s)')
ax.set_ylabel(r'$\mathcal{P} (\text{Rate})$')
ax.yaxis.set_label_coords(-0.2, 0.5)
ax.set_ylim((-0.02, 0.25))
ax.set_xticks([0., 20., 40., 60.])
ax.set_xlim((-10., 60.))
ax.set_yticks([0., 0.1, 0.2])


"""
Save figure
"""
fig.subplots_adjust(left=0.05, right=0.95, top=0.95,
                    bottom=0.075, wspace=1., hspace=.5)
pl.savefig('Fig6_comparison_exp_spiking_data.eps')
