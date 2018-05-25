import correlation_toolbox.helper as ch
import correlation_toolbox.correlation_analysis as corr
import json
import numpy as np
import os
import sys

""" 
Compute the cross-correlation betwen two given areas from their
time series of population-averaged spike rates of a given simulation.
"""

data_path = sys.argv[1]
label = sys.argv[2]
area1 = sys.argv[3]
area2 = sys.argv[4]


load_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'rate_time_series_full')
save_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'cross_correlation')

with open(os.path.join(data_path, label, 'custom_params_{}'.format(label)), 'r') as f:
    sim_params = json.load(f)
T = sim_params['T']

fn1 = os.path.join(load_path,
                   'rate_time_series_full_{}.npy'.format(area1))
rate_time_series1 = np.load(fn1)


fn2 = os.path.join(load_path,
                   'rate_time_series_full_{}.npy'.format(area2))
rate_time_series2 = np.load(fn2)

fn = os.path.join(load_path,
                  'rate_time_series_full_Parameters.json')
with open(fn, 'r') as f:
    params = json.load(f)

i_min = int(500. - params['t_min'])
i_max = int(T - params['t_min'])

rates = [rate_time_series1[i_min:i_max],
         rate_time_series2[i_min:i_max]]

dat = [ch.centralize(rates[0], units=True),
       ch.centralize(rates[1], units=True)]
freq, crossspec = corr.crossspec(dat, 1.)
t, cross = corr.crosscorrfunc(freq, crossspec)

sigma = 2.
time_range = np.arange(-5., 5.)
kernel = 1 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-(time_range ** 2 / (2 * sigma ** 2)))
cross_conv = np.zeros_like(cross)
cross_conv[0][0] = np.convolve(kernel, cross[0][0], mode='same')
cross_conv[0][1] = np.convolve(kernel, cross[0][1], mode='same')
cross_conv[1][0] = np.convolve(kernel, cross[1][0], mode='same')
cross_conv[1][1] = np.convolve(kernel, cross[1][1], mode='same')
cross = cross_conv

fp = '_'.join(('cross_correlation',
               area1,
               area2))
np.save('{}/{}.npy'.format(save_path, fp), cross)
np.save('{}/{}.npy'.format(save_path, 'cross_correlation_time'), t)
