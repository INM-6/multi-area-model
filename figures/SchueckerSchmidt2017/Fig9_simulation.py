import pylab as pl
import os
import numpy as np
from plotfuncs import create_fig
from rate_matrix_plot import rate_matrix_plot, rate_histogram_plot
from area_list import area_list
from multiarea_model.multiarea_helpers import create_vector_mask
from multiarea_model import MultiAreaModel
from plotcolors import myblue, myblue2, myred2, myred
base_dir = os.getcwd()

"""
Figure layout
"""
scale = 1.
width = 5.2
n_horz_panels = 2.
n_vert_panels = 2.
panel_factory = create_fig(1, scale, width,
                           n_horz_panels, n_vert_panels,
                           voffset=0.22, hoffset=0.096, squeeze=0.23)

"""
Simulation with kappa = 1.125 and the stabilized matrix leading to the
improved low-activity fixed point shown in Fig. 9.
"""
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}

sim_params = {'t_sim': 10500.,
              'num_processes': 720,  # Needs to be adapted to the HPC system used
              'local_num_threads': 1,  # Needs to be adapted to the HPC system used
              'recording_dict': {'record_vm': False}}

theory_params = {'T': 30.,
                 'dt': 0.01}


conn_params = {'g': -12.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': 'iteration_4/K_prime.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}
M = MultiAreaModel(network_params,
                   analysis=True,
                   simulation=True,
                   sim_spec=sim_params,
                   theory=True,
                   theory_spec=theory_params)
p, r_stab = M.theory.integrate_siegert()
M.analysis.create_pop_rates()


"""
Panel A
Mean-field theory
"""
ax = panel_factory.new_panel(0, 0, 'A', label_position=-0.25)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('bottom')

cm = pl.get_cmap('rainbow')
cm = cm.from_list('mycmap', [myblue, myblue2,
                             'white', myred2, myred], N=256)


matrix = np.zeros((len(area_list), 8))
for i, area in enumerate(area_list):
    mask = create_vector_mask(M.structure, areas=[area])
    rate = r_stab[:, -1][mask]
    if area == 'TH':
        rate = np.insert(rate, 2, 0.0)
        rate = np.insert(rate, 2, 0.0)
    matrix[i, :] = rate[::-1]

matrix = np.transpose(matrix)
rate_matrix_plot(panel_factory.figure, ax, matrix, position='left')


"""
Panel B
Simulation
"""
ax = panel_factory.new_panel(1, 0, 'B', label_position=-0.25)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('bottom')


matrix_sim = np.zeros((len(M.area_list), 8))
k = 'post'
for i, area in enumerate(M.area_list):
    for j, pop in enumerate(M.structure['V1'][::-1]):
        if pop not in M.structure[area]:
            rate = np.nan
        else:
            rate = M.analysis.pop_rates[area][pop][0]

        if rate == 0.0:
            rate = 1e-5
        matrix_sim[i][j] = rate

matrix_sim = np.transpose(matrix_sim)
rate_matrix_plot(panel_factory.figure, ax, matrix, position='right')


"""
Panel C
Theory vs. simulation
"""

ax = panel_factory.new_panel(0, 1, 'C', label_position=-0.2)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='x', which='minor', length=0.)
ax.tick_params(axis='y', which='minor', length=0.)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.loglog([10 ** -2, 10 ** 2], [10 ** -2, 10 ** 2], color='0.5')
ax.loglog(matrix_sim.flatten(), matrix.flatten(), '.', color='k')
ax.set_xlabel(r'$\nu_\mathrm{simulation} (1/\mathrm{s})$', labelpad=1)
ax.set_ylabel(r'$\nu_\mathrm{theory} (1/\mathrm{s})$', labelpad=-0.5)
ax.set_yticks(10. ** (np.array([-2, 0, 2])))

"""
Panel D
Rate histogram
"""
ax = panel_factory.new_empty_panel(1, 1, 'D', label_position=-0.2)
rate_histogram_plot(panel_factory.figure, ax, matrix_sim, position='left')


"""
Save figure
"""
pl.savefig('Fig9_simulation.eps')
