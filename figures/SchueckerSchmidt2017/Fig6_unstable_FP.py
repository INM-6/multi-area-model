import pylab as pl
import os
import numpy as np
import pylab as pl
import utils

from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_vector_mask
from plotfuncs import create_fig
from plotcolors import myblue, myblue2, myred2, myred
from rate_matrix_plot import rate_matrix_plot
from area_list import area_list

base_dir = os.getcwd()
cmap = pl.cm.rainbow
cmap = cmap.from_list(
    'mycmap', [myblue, myblue2, 'white', myred2, myred], N=256)


"""
Figure layout
"""
scale = 1.
width = 4.8
n_horz_panels = 2.
n_vert_panels = 1.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.19, hoffset=0.096, squeeze=0.23)

"""
Create instance of MultiAreaModel
"""
conn_params = {'g': -16.,
               'av_indegree_V1': 3950.,
               'fac_nu_ext_TH': 1.2}
input_params = {'rate_ext': 8.}

network_params = {'connection_params': conn_params,
                  'input_params': input_params}
theory_params = {'dt': 0.01,
                 'T': 30.}
time = np.arange(0., theory_params['T'], theory_params['dt'])

M_base = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)

"""
Plot data
"""
labels = ['A', 'B']
for ii, label in enumerate(labels):
    ax = panel_factory.new_panel(ii, 0, '' + labels[ii], label_position='leftleft')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')

    data = utils.load_iteration(ii + 1)
    (par_transition, r_low, r_high,
     minima_low, minima_high) = utils.determine_velocity_minima(time, data)

    unstable_low = r_low[:, minima_low[1]]

    matrix = np.zeros((len(area_list), 8))
    for i, area in enumerate(area_list):
        mask = create_vector_mask(M_base.structure, areas=[area])
        m = unstable_low[mask]
        if area == 'TH':
            m = np.insert(m, 2, 0.0)
            m = np.insert(m, 2, 0.0)
        matrix[i, :] = m[::-1]
    matrix = np.transpose(matrix)

    if ii == 0:
        rate_matrix_plot(panel_factory.figure, ax, matrix, position='left')
    else:
        rate_matrix_plot(panel_factory.figure, ax, matrix, position='right')

"""
Save figure
"""
pl.savefig('Fig6_unstable_FP.eps')
