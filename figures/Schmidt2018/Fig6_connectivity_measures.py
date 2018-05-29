import os
import matplotlib.pyplot as pl
import numpy as np
import sys

from helpers import area_list, population_labels
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask, create_vector_mask
from plotcolors import myred
import pyx

if len(sys.argv) > 1:
    plot = sys.argv[1]
else:
    plot = ''


base_dir = os.getcwd()
cmap = pl.cm.coolwarm
cmap2 = cmap.from_list('mycmap', ['white', myred], N=256)
cmap2 = pl.get_cmap('inferno')

M = MultiAreaModel({})

"""
Layout
"""
nrows = 1
ncols = 2
width = 6.8504
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

height = width / panel_wh_ratio * float(nrows) / ncols
height = 6.

pl.rcParams['figure.figsize'] = (width, height)

gs1 = gridspec.GridSpec(1, 2)
gs1.update(left=0.1, right=0.99, top=0.65, wspace=0.1, bottom=0.05)
axes = {}
axes['A'] = pl.subplot(gs1[:-1, :1])
axes['B'] = pl.subplot(gs1[:-1, 1:2])

gs2 = gridspec.GridSpec(1, 2)
gs2.update(left=0.1, right=0.99, top=0.73, wspace=0.1, bottom=0.65)

axes['A2'] = pl.subplot(gs2[:-1, :1], frameon=False)
axes['B2'] = pl.subplot(gs2[:-1, 1:2], frameon=False)
axes['A2'].set_xticks([])
axes['A2'].set_yticks([])
axes['B2'].set_xticks([])
axes['B2'].set_yticks([])

num_neurons = []
for area in area_list:
    for pop in M.structure[area]:
        num_neurons.append(M.N[area][pop])
num_neurons = np.array(num_neurons)

prob = {}

Npre = np.zeros_like(M.K_matrix[:, :-1])
Npost = np.zeros_like(M.K_matrix[:, :-1])

num_vector = np.zeros_like(M.K_matrix[:, :-1][:, 0])
index = 0
for area in area_list:
    for pop in M.structure[area]:
        num_vector[index] = M.N[area][pop]
        index += 1

for i in range(254):
    Npre[i] = num_vector
    Npost[:, i] = num_vector

C = 1. - (1. - 1. / (Npre * Npost))**(M.K_matrix[:, :-1] * Npost)
Nsyn = M.K_matrix[:, :-1] * Npost
outdegree = Nsyn / Npre
indegree = M.K_matrix[:, :-1]

plot_areas = ['V1', 'V2']
mask = create_mask(M.structure, target_areas=plot_areas,
                   source_areas=plot_areas,
                   extern=False)[:, :-1]
vmask = create_vector_mask(M.structure, areas=plot_areas)
new_size = np.where(vmask)[0].size

Nsyn_plot = Nsyn[mask].reshape((new_size, new_size))
C_plot = C[mask].reshape((new_size, new_size))
indegree_plot = indegree[mask].reshape((new_size, new_size))
outdegree_plot = outdegree[mask].reshape((new_size, new_size))

t_index = 0
ticks = []
ticks_r = []
for area in plot_areas:
    ticks.append(t_index + 0.5 * len(M.structure[area]))
    ticks_r.append(new_size - (t_index + 0.5 * len(M.structure[area])))
    for pop in M.structure[area]:
        t_index += 1


ax = axes['A']
ax.set_aspect(1. / ax.get_data_ratio())
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")

matrix = indegree_plot
plot_norm = LogNorm(vmin=0.1, vmax=3000.)
im = ax.pcolormesh(matrix[::-1], cmap=cmap2, norm=plot_norm)
cb = pl.colorbar(im, orientation='horizontal',
                 ax=axes['A2'], fraction=0.6, shrink=0.7)
sm = pl.cm.ScalarMappable(cmap=cmap2, norm=plot_norm)
ax.set_xlabel('Source population')
ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_ylabel('Target population')
ax.yaxis.set_label_coords(-0.15, 0.5)
ax.set_xlim((0, new_size))
ax.set_ylim((0, new_size))

ax.set_xticks(np.arange(16) + 0.5)
ax.set_xticklabels(2 * population_labels, rotation=90, size=7.)

ax.set_yticks(np.arange(16)[::-1] + 0.5)
ax.set_yticklabels(2 * population_labels, size=7.)

ax.text(-2., ticks_r[0], plot_areas[0], size=9., rotation=90)
ax.text(-2., ticks_r[1], plot_areas[1], size=9., rotation=90)

ax.text(ticks[0], -2., plot_areas[0], size=9.)
ax.text(ticks[1], -2., plot_areas[1], size=9.)

pos = cb.ax.get_position()
ax_hist = pl.axes(
    [pos.x0, pos.y0 + 0.06, pos.x1 - pos.x0, pos.y1 - pos.y0])
bins = 10**np.linspace(np.log10(cb.get_clim()
                                [0]), np.log10(cb.get_clim()[1]), 12)
vals, bins = np.histogram(matrix, bins=bins)

ax_hist.spines['right'].set_color('none')
ax_hist.spines['left'].set_color('none')
ax_hist.spines['top'].set_color('none')
ax_hist.yaxis.set_ticks_position("none")
ax_hist.xaxis.set_ticks_position("none")

colors = sm.to_rgba(bins[:-1])
ax_hist.bar(bins[:-1], vals, width=np.diff(bins), color=colors,
            edgecolor='k', align='edge')
ax_hist.set_xscale('Log')
ax_hist.set_xlim(cb.get_clim())
ax_hist.set_xticks([])
ax_hist.set_yticks([])

ax = axes['B']

matrix = C_plot
plot_norm = LogNorm(vmin=0.0001, vmax=np.max(C_plot))
im = ax.pcolormesh(matrix[::-1], cmap=cmap2, norm=plot_norm)
cb = pl.colorbar(im, orientation='horizontal',
                 ax=axes['B2'], fraction=0.6, shrink=0.7)
sm = pl.cm.ScalarMappable(cmap=cmap2, norm=plot_norm)
ax.set_aspect(1. / ax.get_data_ratio())
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")

ax.set_xlim((0, new_size))
ax.set_ylim((0, new_size))

ax.set_xlabel('Source population')
ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_xticks(np.arange(16) + 0.5)
ax.set_xticklabels(2 * population_labels, rotation=90, size=7.)

ax.text(ticks[0], -2., plot_areas[0], size=9.)
ax.text(ticks[1], -2., plot_areas[1], size=9.)

ax.set_yticks([])

pos = cb.ax.get_position()
ax_hist = pl.axes(
    [pos.x0, pos.y0 + 0.06, pos.x1 - pos.x0, pos.y1 - pos.y0])

bins = 10**np.linspace(np.log10(cb.get_clim()
                                [0]), np.log10(cb.get_clim()[1]), 12)
vals, bins = np.histogram(matrix, bins=bins)

ax_hist.spines['right'].set_color('none')
ax_hist.spines['left'].set_color('none')
ax_hist.spines['top'].set_color('none')
ax_hist.yaxis.set_ticks_position("none")
ax_hist.xaxis.set_ticks_position("none")

colors = sm.to_rgba(bins[:-1])
ax_hist.bar(bins[:-1], vals, width=np.diff(bins), color=colors,
            edgecolor='k', align='edge')
ax_hist.set_xscale('Log')
ax_hist.set_xlim(cb.get_clim())
ax_hist.set_xticks([])
ax_hist.set_yticks([])

pl.savefig('Fig6_connectivity_measures_mpl.eps')

c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(
    0., 0.5, "Fig6_connectivity_measures_mpl.eps", width=17.3))
c.insert(pyx.epsfile.epsfile(
    2.1, 12.5, "Fig6_conn_indegree.eps", width=6.5))
c.insert(pyx.epsfile.epsfile(10., 12.5, "Fig6_conn_prob.eps", width=6.5))
c.writeEPSfile("Fig6_connectivity_measures.eps")
