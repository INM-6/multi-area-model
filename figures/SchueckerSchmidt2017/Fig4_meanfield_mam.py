import pyx
import utils
import pylab as pl
import os
from matplotlib.colors import ListedColormap
import matplotlib.hatch
import numpy as np
import matplotlib.pyplot as plt
from plotcolors import myblue, myblue2, myred2, myred
from plotfuncs import create_fig
from area_list import area_list
from rate_matrix_plot import rate_matrix_plot
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_vector_mask
base_dir = os.getcwd()

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
Figure layout
"""
scale = 1.
width = 4.8
n_horz_panels = 2.
n_vert_panels = 3.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.22, hoffset=0.1, squeeze=0.23)

ax = panel_factory.new_empty_panel(
    0, 0, '', label_position=-0.2)
ax = pl.axes(ax, frameon=False)
ax.text(0.4, 1., r'\bfseries{}' + 'A',
        fontdict={'fontsize': 10.,
                  'weight': 'bold',
                  'horizontalalignment': 'left',
                  'verticalalignment': 'bottom'},
        transform=ax.transAxes)

ax_traj = panel_factory.new_panel(0, 1, 'B', label_position=-0.25)
ax_vel = panel_factory.new_panel(1, 1, 'C', label_position=-0.25)

for ax in [ax_traj, ax_vel]:
    ax.set_yscale('Log')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    
theory_params = {'dt': 0.01,
                 'T': 30.}

"""
Parameter Space
"""
data = utils.load_iteration(1)
param_list = data['parameters']
traj = np.mean(data['results'], axis=1)

plot_params = np.append(np.arange(1., 1.1, 0.01),
                        [1.0743549000000001, 1.0743549999999995])
ind = [np.where(param_list == par)[0][0] for par in plot_params]

traj = traj[ind]

"""
Panel B: Plot of trajectories
"""
cmap = plt.get_cmap('binary')
line_colors = cmap(np.linspace(0, 1, len(traj) - 1))
cmap2 = ListedColormap(line_colors)
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(
    vmin=plot_params[0], vmax=plot_params[-3]))
sm.set_array([])

time = np.arange(0., theory_params['T'], theory_params['dt'])
[ax_traj.plot(time,
              traj[i],
              color=line_colors[i]) for i in range(len(traj) - 2)]
ax_traj.plot(time, traj[-2], color=myblue)
ax_traj.plot(time, traj[-1], color=myred)

# Plot colorbar
cbticks = [1., 1.1]
cbar = plt.colorbar(sm, ax=ax, ticks=cbticks, shrink=0.5, aspect=15,
                    pad=-0.0, anchor=(-11., 0.45))
cbar.solids.set_edgecolor('none')
ax_traj.text(23., 20., r'$\kappa$')

ax_traj.set_xlabel(r'Time $s$', labelpad=-0.1)
ax_traj.set_ylabel(
    r'$\langle \boldsymbol{\nu} \rangle \, (1/\mathrm{s})$', labelpad=-0.1)
ax_traj.set_ylim((1., 500.))
ax_traj.set_xlim((0., 26.))
# ax_traj.vlines(time_critical, 1., 350., linestyles='dashed')
# ax_traj.text(time_critical, 400., r'$s_{\mathrm{c}}$')
ax_traj.set_xticks(np.arange(0., 20.1, 10.))
fmt = matplotlib.ticker.LogFormatterMathtext(
    base=10.0, labelOnlyBase=False)
ax_traj.yaxis.set_major_formatter(fmt)
ax_traj.yaxis.set_minor_locator(plt.NullLocator())


"""
Panel C: Flow of trajectories
"""
ax_vel.set_xlabel(r'Time $s$', labelpad=-0.1)
ax_vel.set_ylabel(r'$\| \boldsymbol{\dot{\nu}} (s) \|$', labelpad=-0.05)
ax_vel.set_yscale('Log')
ax_vel.set_xticks(np.arange(0., 20.1, 10.))
ax_vel.set_yticks(10 ** (np.arange(-4., 2.5, 2.)))
ax_vel.set_yticklabels(10 ** (np.arange(-3., -0.5, 1.)))
fmt = matplotlib.ticker.LogFormatterMathtext(
    base=10.0, labelOnlyBase=False)
ax_vel.yaxis.set_major_formatter(fmt)
ax_vel.yaxis.set_minor_locator(plt.NullLocator())
ax_vel.set_ylim((1e-5, 1e2))
ax_vel.set_xlim((0., 21.))

d_nu, minima = utils.velocity_peaks(time, data['results'][ind[-2]], threshold=0.05)
ax_vel.plot(time[:-1], d_nu, color=myblue)
ax.vlines(time[minima[-1]], 1e-5, 1e0, linestyles='dashed', color='k')
ax_traj.vlines(time[minima[-1]], 1e0, 1e2, linestyles='dashed', color='k')

d_nu, minima = utils.velocity_peaks(time, data['results'][ind[-1]], threshold=0.05)
ax_vel.plot(time[:-1], d_nu, color=myred)

"""
Panels C,D
"""

labels = ['D', 'E']
cm = pl.get_cmap('rainbow')
cm = cm.from_list('mycmap', [myblue, myblue2,
                             'white', myred2, myred], N=256)
par_list = [1., 1.125]
for ii, par in enumerate(par_list):
    ind = np.where(param_list == par)[0][0]
    ax = panel_factory.new_panel(ii, 2, labels[ii], label_position=-0.31)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')

    matrix = np.zeros((len(area_list), 8))
    
    for i, area in enumerate(area_list):
        mask = create_vector_mask(M_base.structure, areas=[area])
        rate = data['results'][ind][:, -1][mask]
        if area == 'TH':
            rate = np.insert(rate, 2, 0.0)
            rate = np.insert(rate, 2, 0.0)
        matrix[i, :] = rate[::-1]

    matrix = np.transpose(matrix)
    if ii == 0:
        rate_matrix_plot(panel_factory.figure, ax,
                         matrix, position='left')
    else:
        rate_matrix_plot(panel_factory.figure, ax,
                         matrix, position='right')

"""
Save figure
"""
print(base_dir)
pl.savefig('Fig4_meanfield_mam_mpl.eps')

"""
Merge figures
"""
c = pyx.canvas.canvas()

c.insert(pyx.epsfile.epsfile(0, 0, "Fig4_meanfield_mam_mpl.eps"))
c.insert(pyx.epsfile.epsfile(3., 7.5, "separatrix.eps", width=6.))

c.writeEPSfile("Fig4_meanfield_mam.eps")
