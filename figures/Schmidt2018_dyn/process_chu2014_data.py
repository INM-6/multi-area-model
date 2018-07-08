from helpers import chu2014_path
import numpy as np
import os
import scipy.io
import sys

from multiarea_model.analysis_helpers import centralize
from scipy.signal import spectrogram, welch

data_path = sys.argv[1]
sim_label = sys.argv[2]

"""
Load data
"""

data_pvc5 = {'ids': [], 'times': []}
id_neuron = 0
id_to_channel = {}
channel_depths = {}

# Information in crcns-pvc5-data-description.pdf
for channel in range(65, 129):
    depth = (channel - 65) % 8
    channel_depths[channel] = depth


load_path = os.path.join(chu2014_path,
                         'crcns-pvc5/rawSpikeTime')
save_path = os.path.join(chu2014_path,
                         'Analysis')

try:
    os.mkdir(save_path)
except FileExistsError:
    pass

for fn in os.listdir(load_path):
    # each file contains several neurons
    temp = scipy.io.loadmat(os.path.join(load_path, fn))['cluster_class']
    channel = fn.split('_')[-1].split('.')[0][2:]
    ids = set(temp[:, 0])
    for id_temp in ids:
        mask = temp[:, 0] == id_temp
        data_pvc5['ids'] += [id_neuron]
        data_pvc5['times'] += [temp[:, 1][mask]]
        id_to_channel[id_neuron] = channel
        id_neuron += 1

neuron_depths = [channel_depths[int(id_to_channel[id])]
                 for id in range(0, 140)]

"""
Take only neurons of the first 113 electrodes into account which are
less than 1 mm apart
"""
ind_mm = []
for i, id in enumerate(data_pvc5['ids']):
    if int(id_to_channel[id]) <= 112:
        ind_mm.append(i)

np.save(os.path.join(save_path,
                     'spike_data_1mm.npy'),
        np.array(data_pvc5['times'])[ind_mm])

np.save(os.path.join(save_path,
                     'neuron_depths.npy'),
        np.array(neuron_depths)[ind_mm])

t_min = np.min([np.min(x) for x in np.array(data_pvc5['times'])[ind_mm]])
t_max = np.max([np.max(x) for x in np.array(data_pvc5['times'])[ind_mm]])

time = np.arange(t_min, t_max + 1., 1.)
rate_binned = np.array([np.histogram(x, time)[0] for x in
                        np.array(data_pvc5['times'])[ind_mm]]) * 1e3
time = time[:-1]
pop_rate = np.mean(rate_binned, axis=0)

np.save(os.path.join(save_path,
                     'rate_time_series_V1.npy'),
        rate_binned)
np.save(os.path.join(save_path,
                     'rate_time_series_time.npy'),
        time)

# Parameters for Welch Power Spectral density and spectrogram
noverlap = 1000
nperseg = 1024
window = 'boxcar'
fs = 1.e3


"""
Compute spectrogram for panel A
"""
pop_rate = centralize(pop_rate, units=True)
f, t, Sxx = spectrogram(pop_rate, fs=fs, nperseg=4096,
                        noverlap=0, window=window)

np.save(os.path.join(save_path,
                     'spectrogram_time.npy'),
        t)
np.save(os.path.join(save_path,
                     'spectrogram_freq.npy'),
        f)
np.save(os.path.join(save_path,
                     'spectrogram_Sxx.npy'),
        Sxx)

"""
Detect phases of low and high fluctuations:
Criterium: Integrated power for f in [0, 40] Hz
See Methods section in the paper.
"""
f_crit = 40.
ind = np.where(f <= f_crit)[0]
threshold = 125.
ind_raw = np.where(f <= f_crit)[0]
Sxx_int = np.sum(Sxx, axis=0)
Sxx_int_low = np.sum(Sxx[ind], axis=0)
ind_low = np.where(Sxx_int_low <= threshold)[0]
ind_high = np.where(Sxx_int_low > threshold)[0]


"""
Compute power spectra
"""
time_ind_low = []
for ti in t[ind_low]:
    time_ind_low += list(np.where(np.logical_and(time > 1e3*(ti - 5.),
                                                 time <= 1e3*(ti+5.)))[0])

time_ind_high = []
for ti in t[ind_high]:
    time_ind_high += list(np.where(np.logical_and(time > 1e3*(ti - 5.),
                                                  time <= 1e3*(ti+5.)))[0])

for phase, times in zip(['low_fluct', 'high_fluct', 'full'],
                        [time_ind_low, time_ind_high, None]):
    if times is not None:
        rate = pop_rate[times]
    else:
        rate = pop_rate
    rate = centralize(rate, units=True)
    f, power_fluct = welch(rate,
                           fs=fs,
                           noverlap=noverlap,
                           nperseg=nperseg,
                           window=window)
    np.save(os.path.join(save_path,
                         'power_spectrum_freq.npy'),
            f)
    np.save(os.path.join(save_path,
                         'power_spectrum_V1_{}.npy'.format(phase)),
            power_fluct)
"""
Compute rate histograms
"""
# Choose the same bins used for the analysis of the simulation
fn = os.path.join(data_path, sim_label,
                  'Analysis',
                  'rate_histogram',
                  'rate_histogram_{}.npy'.format('bins'))
bins = np.load(fn)

rate_binned_low = rate_binned[:, time_ind_low]
vals_low, bins = np.histogram(
    np.mean(rate_binned_low, axis=1), bins=bins)

rate_binned_high = rate_binned[:, time_ind_high]
vals_high, bins = np.histogram(
    np.mean(rate_binned_high, axis=1), bins=bins)

vals_full, bins = np.histogram(np.mean(rate_binned, axis=1), bins=bins)

np.save(os.path.join(save_path,
                     'rate_histogram_bins.npy'),
        bins)

np.save(os.path.join(save_path,
                     'rate_histogram_low.npy'),
        vals_low)
np.save(os.path.join(save_path,
                     'rate_histogram_high.npy'),
        vals_high)
np.save(os.path.join(save_path,
                     'rate_histogram_full.npy'),
        vals_full)

