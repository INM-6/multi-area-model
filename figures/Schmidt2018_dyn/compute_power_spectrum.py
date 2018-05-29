import numpy as np
import os
import sys

from multiarea_model import MultiAreaModel
from multiarea_model.analysis_helpers import centralize
from scipy.signal import welch


"""
Compute the power spectrum time series for a given area of
population-averaged spike rates of a given simulation.

The spike rates can be based on three different methods:
- binned spike histograms on all neurons ('full')
- binned spike histograms on a subsample of 140 neurons ('subsample')
- spike histograms convolved with a Gaussian kernel of optimal width
  after Shimazaki et al. (2010)
"""

# Parameters for Welch Power Spectral density and spectrogram
noverlap = 1000
nperseg = 1024
window = 'boxcar'
fs = 1.e3

assert(len(sys.argv) == 5)
data_path = sys.argv[1]
label = sys.argv[2]
area = sys.argv[3]
method = sys.argv[4]

load_dir = os.path.join(data_path,
                        label,
                        'Analysis')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'power_spectrum_{}'.format(method))


assert(method in ['subsample', 'full', 'auto_kernel'])
# subsample : subsample spike data to 140 neurons to match the Chu 2014 data
# full : use spikes of all neurons and compute spike histogram with bin size 1 ms
# auto_kernel : use spikes of all neurons and convolve with Gaussian
#               kernel of optimal width using the method of Shimazaki et al. (2012)
#               (see Method parts of the paper)

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

fn = os.path.join(load_dir,
                  'rate_time_series_{}'.format(method),
                  'rate_time_series_{}_{}.npy'.format(method, area))
rate_time_series = np.load(fn)

rate_centr = centralize(rate_time_series, units=True)
freq, power_subsampled = welch(rate_centr, fs=fs,
                               noverlap=noverlap, nperseg=nperseg)

fn = os.path.join(save_path,
                  'power_spectrum_{}_{}.npy'.format(method, 'freq'))
np.save(fn, freq)

fn = os.path.join(save_path,
                  'power_spectrum_{}_{}.npy'.format(method, area))
np.save(fn, power_subsampled)
