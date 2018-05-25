import correlation_toolbox.helper as ch
import numpy as np
import os
import sys

from multiarea_model import MultiAreaModel
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

"""
Compute the functional connectivity between all areas of a given
simulation based on their time series of spiking rates or their
estimated BOLD signal.
"""

data_path = sys.argv[1]
label = sys.argv[2]
method = sys.argv[3]

load_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         method)
save_path = os.path.join(data_path,
                         label,
                         'Analysis')

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

time_series = []
for area in M.area_list:
    fn = os.path.join(load_path,
                      '{}_{}.npy'.format(method, area))
    si = np.load(fn)
    if method == 'bold_signal':  # Cut off the long initial transient of the BOLD signal
        si = si[5000:]
    time_series.append(ch.centralize(si, units=True))

D = pdist(time_series, metric='correlation')
correlation_matrix = 1. - squareform(D)

np.save(os.path.join(save_path,
                     'functional_connectivity_{}.npy'.format(method)),
        correlation_matrix)
