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
load_path = os.getenv('HOME') + '/datasets_USB/datasets/Simulations/data_dynamics_manuscript/'
data = {}


os.chdir(os.path.join('sim_Model1B_533d73357fbe99f6178029e6054b571b485f40f6'))
with open('Analysis/pop_rates.json', 'r') as f:
    data['LA'] = json.load(f)

os.chdir(os.path.join('sim_Model1B_0adda4a542c3d5d43aebf7c30d876b6c5fd1d63e'))
with open('Analysis/pop_rates.json', 'r') as f:
    data['HA'] = json.load(f)


labels_top = ['C', 'D']
labels_bottom = ['E', 'F']

M = MultiAreaModel({})

"""
Plot data of LA and HA state using
plot functions define rate_matrix_plot.py
"""
for ii, k in enumerate(['LA', 'HA']):
    ax = panel_factory.new_panel(
        ii, 2, labels_top[ii], label_position=-0.25)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')

    matrix = np.zeros((len(area_list), 8))

    for i, area in enumerate(area_list):
        for j, pop in enumerate(M.structure['V1'][::-1]):
            if pop not in M.structure[area]:
                rate = np.nan
            else:
                rate = data[k][area][pop][0]

            if rate == 0.0:
                rate = 1e-5
            matrix[i][j] = rate

    matrix = np.transpose(matrix)
    ax2 = panel_factory.new_empty_panel(
        ii, 3, labels_bottom[ii], label_position=-0.2)

    if ii == 0:
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
pyx.text.set(mode='latex')
pyx.text.preamble(r"\usepackage{helvet}")

c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0, "Fig3_bistability_mpl.eps"))

c.insert(pyx.epsfile.epsfile(0.3, 8., "Microcircuit_sketch.eps", width=4.))
c.insert(pyx.epsfile.epsfile(5., 8.5, "MAM_sketch.eps", width=6.5))

c.writeEPSfile("Fig3_bistability.eps")
