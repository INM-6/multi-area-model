import json
import matplotlib.pyplot as pl
import numpy as np
import os
import pyx

from helpers import original_data_path
from plotfuncs import create_fig
from matrix_plot import matrix_plot, rate_histogram_plot
from multiarea_model import MultiAreaModel

LOAD_ORIGINAL_DATA = True

scale = 1.
width = 7.0866
n_horz_panels = 3.
n_vert_panels = 3.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.25, hoffset=0.1, squeeze=0.1)

axes = {}
axes['A'] = panel_factory.new_panel(0, 1, r'A', label_position=-0.25)
axes['A2'] = panel_factory.new_empty_panel(0, 2, r'', label_position=-0.25)

axes['B'] = panel_factory.new_panel(1, 1, r'B', label_position=-0.25)
axes['B2'] = panel_factory.new_empty_panel(1, 2, r'', label_position=-0.25)

axes['C'] = panel_factory.new_panel(2, 1, r'C', label_position=-0.25)
axes['C2'] = panel_factory.new_empty_panel(2, 2, r'', label_position=-0.25)

# Simulation
if LOAD_ORIGINAL_DATA:
    data = {}
    data_labels = [('LA', '533d73357fbe99f6178029e6054b571b485f40f6'),
                   ('HA', '0adda4a542c3d5d43aebf7c30d876b6c5fd1d63e'),
                   ('LA_post', '33fb5955558ba8bb15a3fdce49dfd914682ef3ea')]
    for key, label in data_labels:
        fn = os.path.join(original_data_path, label, 'Analysis/pop_rates.json')
        with open(fn, 'r') as f:
            data[key] = json.load(f)

    """
    Create MultiAreaModel instance to have access to data structures
    """
    M = MultiAreaModel({})


labels = ['A', 'B', 'C']

for ii, k in enumerate(['LA', 'HA', 'LA_post']):
    ax = axes[labels[ii]]
    ax2 = axes[labels[ii] + '2']
    print(k)
    matrix = np.zeros((len(M.area_list), 8))

    for i, area in enumerate(M.area_list):
        for j, pop in enumerate(M.structure['V1'][::-1]):
            if pop not in M.structure[area]:
                rate = np.nan
            else:
                rate = data[k][area][pop][0]

            if rate == 0.0:
                rate = 1e-5
            matrix[i][j] = rate

    matrix = np.transpose(matrix)

    if ii == 0:
        matrix_plot(panel_factory.figure, ax, matrix, position='left')
        rate_histogram_plot(panel_factory.figure, ax2,
                            matrix, position='left')
    elif ii == 1:
        matrix_plot(panel_factory.figure, ax, matrix, position='center')
        rate_histogram_plot(panel_factory.figure, ax2,
                            matrix, position='center')
    else:
        matrix_plot(panel_factory.figure, ax, matrix, position='right')
        rate_histogram_plot(panel_factory.figure, ax2,
                            matrix, position='right')

pl.savefig('Fig2_bistability_mpl.eps')

"""
Merging files
"""
pyx.text.set(cls=pyx.text.LatexRunner)
pyx.text.preamble(r"\usepackage{helvet}")

c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0.5, 0.5, "Fig2_bistability_mpl.eps", width=17.6))
c.insert(pyx.epsfile.epsfile(
    4., 8.5, "phasespace_sketch.eps", width=10.))
c.insert(pyx.epsfile.epsfile(1., 3.1, "Epop.eps", width=0.75))
c.insert(pyx.epsfile.epsfile(1., 2., "Ipop.eps", width=0.75))


c.writeEPSfile("Fig2_bistability.eps")
