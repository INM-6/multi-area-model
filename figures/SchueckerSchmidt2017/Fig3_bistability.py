import h5py_wrapper.wrapper as h5
import json
import matplotlib.pyplot as pl
import numpy as np
import os
import pyx
import sys

from plotfuncs import create_fig
from rate_matrix_plot import rate_matrix_plot, rate_histogram_plot
from area_list import area_list
from multiarea_model import MultiAreaModel

base_dir = os.getcwd()

"""
Figure layout
"""
scale = 1.
width = 4.566929  # inches for 1.5 JoN columns
n_horz_panels = 2.
n_vert_panels = 4.
panel_factory = create_fig(1, scale, width, n_horz_panels,
                           n_vert_panels, voffset=0.22, hoffset=0.1, squeeze=0.23)

ax = panel_factory.new_empty_panel(
    0, 0, 'A', label_position=-0.19)

ax = panel_factory.new_empty_panel(
    1, 0, 'B', label_position=-0.2)

"""
Load data
"""

# Common parameter settings
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}

sim_params = {'t_sim': 10500.,
              'num_processes': 720,  # Needs to be adapted to the HPC system used
              'local_num_threads': 1,  # Needs to be adapted to the HPC system used
              'recording_dict': {'record_vm': False}}

theory_params = {'T': 30.,
                 'dt': 0.01}

"""
Simulation with kappa = 1. leading to the low-activity fixed point
shown in Fig. 4D.
"""
d = {}
conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.,
               'fac_nu_ext_6E': 1.,
               'av_indegree_V1': 3950.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params,
                  'input_params': input_params}
M_LA = MultiAreaModel(network_params,
                      simulation=True,
                      sim_spec=sim_params,
                      analysis=True)
M_LA.analysis.create_pop_rates()

"""
Simulation with kappa = 1.125 leading to the high-activity fixed point
shown in Fig. 4E.
"""
conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

M_HA = MultiAreaModel(network_params, simulation=True,
                      sim_spec=sim_params,
                      analysis=True)
M_HA.analysis.create_pop_rates()

data = {'LA': M_LA.pop_rates,
        'HA': M_HA.pop_rates}

"""
Plot data of LA and HA state using
plot functions define rate_matrix_plot.py
"""
labels_top = ['C', 'D']
labels_bottom = ['E', 'F']

for i, k in enumerate(['LA', 'HA']):
    ax = panel_factory.new_panel(
        i, 2, labels_top[i], label_position=-0.25)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')

    matrix = np.zeros((len(area_list), 8))

    for i, area in enumerate(area_list):
        for j, pop in enumerate(M_LA.structure['V1'][::-1]):
            if pop not in M_LA.structure[area]:
                rate = np.nan
            else:
                rate = data[k][area][pop][0]

            if rate == 0.0:
                rate = 1e-5
            matrix[i][j] = rate

    matrix = np.transpose(matrix)
    ax2 = panel_factory.new_empty_panel(
        i, 3, labels_bottom[i], label_position=-0.2)

    if i == 0:
        rate_matrix_plot(panel_factory.figure, ax, matrix, position='left')
        rate_histogram_plot(panel_factory.figure, ax2,
                            matrix, position='left')
    else:
        rate_matrix_plot(panel_factory.figure, ax,
                         matrix, position='right')
        rate_histogram_plot(panel_factory.figure, ax2,
                            matrix, position='right')

"""
Save figure
"""
os.chdir(base_dir)

pl.savefig(os.path.join(base_dir,
                        'Fig3_bistability_mpl.eps'))

"""
Merge figures
"""
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0, "Fig3_bistability_mpl.eps"))

c.insert(pyx.epsfile.epsfile(0.3, 8., "Microcircuit_sketch.eps", width=4.))
c.insert(pyx.epsfile.epsfile(5., 8.5, "MAM_sketch.eps", width=6.5))

c.writeEPSfile("Fig3_bistability.eps")
