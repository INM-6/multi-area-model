import copy
import json
import numpy as np
import pyx
import os

from helpers import original_data_path, population_labels
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import vector_to_dict, create_vector_mask
from plotcolors import myblue, myred
from scipy.signal import find_peaks_cwt
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as pl
from matplotlib import gridspec
from matplotlib import rc_file
rc_file('plotstyle.rc')

cmap = pl.cm.coolwarm
cmap = cmap.from_list('mycmap', [myblue, 'white', myred], N=256)

"""
Figure layout
"""
width = 7.0866
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

nrows = 3
ncols = 3
height = width / panel_wh_ratio * float(nrows) / ncols
height = 6.375
pl.rcParams['figure.figsize'] = (width, height)

fig = pl.figure()
axes = {}


gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.055, right=0.9225, top=0.95,
           bottom=0.07, wspace=0.4, hspace=0.4)

axes['A'] = pl.subplot(gs1[0, 0])
axes['B'] = pl.subplot(gs1[0, 1])
axes['C'] = pl.subplot(gs1[0, 2])
axes['D'] = pl.subplot(gs1[1, 0])
axes['E'] = pl.subplot(gs1[1, 1])
axes['F'] = pl.subplot(gs1[1, 2])

axes['G'] = pl.subplot(gs1[2, 0])
pos = axes['G'].get_position()
axes['G2'] = pl.axes([pos.x1 - 0.08 + 0.5, pos.y0+0.05,
                      0.1,
                      pos.y1 - pos.y0])


fd = {'fontsize': 10, 'weight': 'bold', 'horizontalalignment':
      'left', 'verticalalignment': 'bottom'}


for label in ['C', 'D', 'E', 'F']:
    label_pos = [-0.15, 1.04]
    pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
            fontdict=fd, transform=axes[label].transAxes)

pl.text(-0.15, 1.0, r'\bfseries{}' + 'A',
        fontdict=fd, transform=axes['A'].transAxes)

pl.text(-0.05, 1.0, r'\bfseries{}' + 'G',
        fontdict=fd, transform=axes['G'].transAxes)

"""
Load data
"""

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

LOAD_ORIGINAL_DATA = True
if LOAD_ORIGINAL_DATA:
    label = '99c0024eacc275d13f719afd59357f7d12f02b77'
    data_path = original_data_path
else:
    from network_simulations import init_models
    from config import data_path
    models = init_models('Fig7')
    label = models[0].simulation.label


rate_time_series = {}
for area in M.area_list:
    fn = os.path.join(data_path, label,
                      'Analysis',
                      'rate_time_series_full',
                      'rate_time_series_full_{}.npy'.format(area))
    rate_time_series[area] = np.load(fn)

fn = os.path.join(data_path, label,
                  'Analysis',
                  'rate_time_series_full',
                  'rate_time_series_full_Parameters.json')
with open(fn, 'r') as f:
    rate_time_series['Parameters'] = json.load(f)

cross_correlation = {}
for area in M.area_list:
    cross_correlation[area] = {}
    for area2 in M.area_list:
        fn = os.path.join(data_path, label,
                          'Analysis',
                          'cross_correlation',
                          'cross_correlation_{}_{}.npy'.format(area, area2))
        cross_correlation[area][area2] = np.load(fn)

fn = os.path.join(data_path, label,
                  'Analysis',
                  'cross_correlation',
                  'cross_correlation_time.npy')
cross_correlation['time'] = np.load(fn)


# Correlation peak
def correlation_peak(area, area2, t_min, t_max, width=[5.], max_lag=100.):
    t = cross_correlation['time']
    cross = cross_correlation[area][area2]

    # First look for maxima in cc
    cc = cross[0][1]
    indices = np.where(np.logical_and(t > -max_lag, t < max_lag))
    maxima = find_peaks_cwt(cc[indices], np.array(width))
    max_times = t[indices][maxima]
    if len(maxima) > 0:
        if area == area2:
            selected_max = np.where(t[indices] == 0)[0]
            selected_max_time = 0.
        else:
            selected_max = maxima[np.argmax(cc[indices][maxima])]
            selected_max_time = max_times[np.argmax(cc[indices][maxima])]
        selected_max_value = cc[indices][selected_max]
    else:
        selected_max = np.nan
        selected_max_time = np.nan
        selected_max_value = 0.

    # 2nd, look for minima in cc, i.e., maxima in -cc
    cc = -1. * cross[0][1]
    minima = find_peaks_cwt(cc[indices], np.array(width))
    min_times = t[indices][minima]
    if len(minima) > 0:
        if area == area2:
            selected_min = np.where(t[indices] == 0)[0]
            selected_min_time = 0.
        else:
            selected_min = minima[np.argmax(cc[indices][minima])]
            selected_min_time = min_times[np.argmax(cc[indices][minima])]
        selected_min_value = cc[indices][selected_min]
    else:
        selected_min = np.nan
        selected_min_time = np.nan
        selected_min_value = 0.

    # Then select the one with the larger absolute value
    if abs(selected_max_value) > abs(selected_min_value):
        selected_peak = selected_max
        selected_peak_time = selected_max_time
    else:
        selected_peak = selected_min
        selected_peak_time = selected_min_time
 
    # selected_peak = selected_max
    # selected_peak_time = selected_max_time
    cc = cross[0][1]
    if area != 'MDP' and area2 != 'MDP':
        return selected_peak_time, cc[indices][selected_peak]
    else:
        return np.nan, np.nan


"""
Plotting
"""

"""
Panel A: Matrix plot of rate time series
"""
interval = (10750., 11250.)
x_ticks = np.array([10800, 11200])

tmin = interval[0]
tmax = interval[1]

i_min = int(tmin - rate_time_series['Parameters']['t_min'])
i_max = int(tmax - rate_time_series['Parameters']['t_min'])

areas = []
transitions = []
for area in M.area_list:
    rate = rate_time_series[area][i_min:i_max]
    indices = np.where(rate > 2. * np.mean(rate))
    if len(indices[0]) > 0:
        transitions.append(indices[0][0])
    else:
        transitions.append(rate.size - 1)
    areas.append(area)
areas = np.array(areas)
areas = areas[np.argsort(transitions)]
area_string_A = areas[0]
for area in areas[1:]:
    area_string_A += ' '
    area_string_A += area

transitions = np.sort(transitions)


matrix = np.array([])
time_interval = int(tmax - tmin)
for area in areas:
    rate = rate_time_series[area][i_min:i_max]
    matrix = np.append(matrix, rate / np.mean(rate))
matrix = matrix.reshape((len(areas), time_interval))


y_index = list(range(len(areas)))
y_index = [(a + 0.5) for a in y_index]
ytick_labels = areas
pl.yticks(y_index, ytick_labels)


ax1 = axes['A']
ax1.yaxis.set_ticks_position("left")
ax1.xaxis.set_ticks_position("bottom")
ax1.tick_params(axis='y', length=0.)

x_index = [np.where(np.arange(tmin, tmax, 1.) == i)[0][0] for i in x_ticks]
ax1.set_xticks(x_index)
ax1.set_xticklabels(x_ticks)

ax1.set_ylabel('Area')
ax1.yaxis.set_label_coords(-0.18, 0.5)
ax1.set_yticks(np.arange(32.) + 0.5)
# ax1.set_yticklabels(areas, size=8)
ax1.set_yticks([])
ax1.set_xlabel('Time (ms)', labelpad=-0.05)
im = ax1.pcolormesh(matrix, cmap=pl.get_cmap('inferno'), vmin=0., vmax=12.)
pl.colorbar(im, ax=ax1, ticks=[0., 10., 20.])
ax1.set_ylim((0., 32.))
ax1.text(1.3, 0.65, r'$\nu (\mathrm{spikes/s})$',
         rotation=90, transform=ax1.transAxes)


# ################### PANEL B ####################
"""
Panel B: Cross-correlation of 3 pairs of areas
"""

ax2 = axes['B']
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.yaxis.set_ticks_position("left")
ax2.xaxis.set_ticks_position("bottom")
pl.text(-0.1, 1.0, r'\bfseries{}' + 'B',
        fontdict=fd, transform=ax2.transAxes)

area = 'V1'
areas2 = [area, 'V2', 'FEF']
colors = ['0.0', '0.4', '0.6']

for i, area2 in enumerate(areas2):
    t = cross_correlation['time']
    cross = cross_correlation[area][area2]
    cc = cross[0][1]
    tp, pp = correlation_peak(
        area, area2, 500., 100500., max_lag=100.)

    ax2.plot(t, cc, color=colors[i])
    ax2.vlines(tp, -5000., pp, color=colors[i], linestyle='dashed')

ax2.set_xticks([-100., -50., 0., 50., 100.])
ax2.set_xlim((-100., 100.))
ax2.set_yticks([0., 20000., 100000.])
ax2.set_yticklabels([r'$0.$', r'$2\cdot10^4$', r'$10^5$'])
ax2.set_ylim((-5000., 35000.))
ax2.text(-130., 15000., r'$C(\tau)$', rotation=90)
ax2.set_xlabel(r'Time lag $\tau$ ($\mathrm{ms}$)', labelpad=-0.05)


print("Constructing CC Matrix")
cc_matrix = []
peak_matrix = []

cc_matrix = np.zeros((32, 32))
for i, area in enumerate(M.area_list):
    cc_list = []
    peak_list = []
    for j, area2 in enumerate(M.area_list):
        if cc_matrix[j][i] != 0.:
            cc_matrix[i][j] = -1. * cc_matrix[j][i]
        else:
            tp, peak = correlation_peak(area, area2,
                                        500., 100500., max_lag=100.)
            cc_matrix[i][j] = tp
            peak_list.append(peak)
    peak_matrix.append(peak_list)

peak_matrix = np.array(peak_matrix)
d = {'matrix': cc_matrix}

cc_matrix_masked = np.ma.masked_where(np.isnan(cc_matrix), cc_matrix)

"""
Panel C: Extremum matrix unsorted
"""
ax = axes['C']
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")

ax.set_xlabel('Area B')
# ax.xaxis.set_label_coords(0.5, -0.2)

ax.set_ylabel('Area A')
ax.set_xlim((0, 32))
ax.set_ylim((0, 32))

ax.set_aspect(1. / ax.get_data_ratio())
vlim = np.max(np.abs(cc_matrix_masked))
cmap.set_bad('0.5')
im = ax.pcolormesh(cc_matrix_masked[::-1], cmap=cmap, vmin=-vlim, vmax=vlim)

area_string_C = M.area_list[0]
for area in M.area_list[1:]:
    area_string_C += ' '
    area_string_C += area

ax.set_xticks([])
ax.set_yticks([])

pl.colorbar(im, ax=ax, fraction=0.044, ticks=[-80, -40, 0, 40, 80])
ax.text(44.8, 27., r'Extremum time ($\mathrm{ms}$)', rotation=90)


def dev(i, j, hierarchy, cc_matrix):
    """
    Deviation function for the linear programming algorithm
    determining the hierarchy.
    """
    return (hierarchy[i] - hierarchy[j] - cc_matrix[i][j])


def hier_dev(hierarchy, cc_matrix):
    deviation = 0.
    for i in range(hierarchy.size):
        for j in range(hierarchy.size):
            deviation += (dev(i, j, hierarchy, cc_matrix)) ** 2
    return np.sqrt(deviation)


def create_hierarchy(cc_matrix, areas):
    """
    Determined the heirarchy for a given set of areas and their
    cross-correlation peak matrix.
    """
    res = minimize(hier_dev, np.random.rand(
        cc_matrix[0].size), args=(cc_matrix,))
    hierarchy = res['x']
    index_transformation = np.argsort(hierarchy)
    hierarchical_areas = copy.copy(areas)
    hierarchical_areas = np.array(hierarchical_areas)
    hierarchical_areas = hierarchical_areas[index_transformation]
    hierarchy = hierarchy[index_transformation]
    # Map hierarchy onto [0,1] interval
    hierarchy -= np.min(hierarchy)
    hierarchy /= np.max(hierarchy)
    return res, hierarchy, hierarchical_areas, index_transformation


def count_violations(hierarchy, cc_matrix):
    """
    Count the violations of a given hierarchy based on the given
    matrix of cross-correlation peaks.
    """

    violations = 0
    for i in range(hierarchy.size):
        for j in range(hierarchy.size):
            x1 = hierarchy[i] - hierarchy[j]
            x2 = cc_matrix[i][j]
            if np.sign(x1) != np.sign(x2):
                violations += 1

    return violations / 2.  # Divide by two to not count each pair double


print("Computing hierarchy")
area_list = np.array(M.area_list)
# We exclude area MDP because it does not receive connections from any
# other area and thus does not participate in cortico-cortical
# communication
ind_without_MDP = np.isfinite(cc_matrix[0])
cc_matrix_without_MDP = cc_matrix[ind_without_MDP][:, ind_without_MDP]
area_list_without_MDP = area_list[ind_without_MDP]
res, hierarchy, hierarchical_areas, index_transformation = create_hierarchy(
    cc_matrix_without_MDP, area_list_without_MDP)
hierarchy_to_area_list = []

for area in area_list:
    if area not in ['MDP']:
        hierarchy_to_area_list.append(
            np.where(hierarchical_areas == area)[0][0])

# Export hierarchy to csv
with open('Fig7_temporal_hierarchy.csv', 'w') as f:
    for hier, area in zip(hierarchy, hierarchical_areas):
        f.write(area + ',' + str(hier) + '\n')


"""
Panel D: Extremum matrix sorted
"""

ax = axes['D']
cc_matrix_hier = cc_matrix_without_MDP[index_transformation][:,
                                                             index_transformation]
# Add area MDP to the matrix for plotting purposes
cc_matrix_hier = np.insert(cc_matrix_hier, 0, cc_matrix[14][:-1], axis=0)
cc_matrix_hier = np.insert(cc_matrix_hier, 0, cc_matrix[14], axis=1)

cc_matrix_hier_masked = np.ma.masked_where(
    np.isnan(cc_matrix_hier), cc_matrix_hier)
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")
ax.set_xlabel('Area B')
# ax.xaxis.set_label_coords(0.5, -0.2)
ax.set_ylabel('Area A')
# ax.yaxis.set_label_coords(-0.17, 0.5)


ax.set_xlim((0, 32))
ax.set_ylim((0, 32))

ax.set_aspect(1. / ax.get_data_ratio())
im = ax.pcolormesh(
    cc_matrix_hier_masked[:, ::-1], cmap=cmap, vmin=-vlim, vmax=vlim)

area_string_D = hierarchical_areas[0]
for area in hierarchical_areas[1:]:
    area_string_D += ' '
    area_string_D += area

ax.set_xticks([])
ax.set_yticks([])

pl.colorbar(im, ax=ax, fraction=0.044, ticks=[-80, -40, 0, 40, 80])
ax.text(43.3, 27., r'Extremum time ($\mathrm{ms}$)', rotation=90)

"""
Eigenvalue spectrum and eigenvector projection
"""
conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
               'g': -11.,
               'K_stable': '../../K_stable.npy',
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'cc_weights_I_factor': 2.,
               'cc_weights_factor': 1.9,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'connection_params': conn_params,
                  'neuron_params': neuron_params}

theory_params = {'T': 50.,
                 'dt': 0.1,
                 'initial_rates': 'random_uniform',
                 'initial_rates_iter': 15}

M = MultiAreaModel(network_params, theory=True, simulation=False,
                   theory_spec=theory_params)
pops, rates_full = M.theory.integrate_siegert()
# Here, pick a calculation that converges to the LA state
ana_rates = rates_full[12][:, -1]
lambda_max, slope, slope_sigma, G, EV = M.theory.lambda_max(ana_rates, full_output=True)

"""
Panel E: Eigenvalues
"""
ax = axes['E']
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

pos = ax.get_position()

# Real part < 0
ax0 = pl.axes([pos.x0 + 0.02,
               pos.y0,
               0.6 * (pos.x1 - pos.x0),
               pos.y1 - pos.y0])
ax0.spines['right'].set_color('none')
ax0.spines['top'].set_color('none')

ax0.plot(np.real(EV[0]),
         np.imag(EV[0]), '.')

ax0.set_xlabel(r'$\mathrm{Re}(\lambda_i)$')
ax0.xaxis.set_label_coords(1., -0.2)
ax0.yaxis.set_label_coords(-0.15, 0.5)
ax0.set_ylabel(r'$\mathrm{Im}(\lambda_i)$')
ax0.yaxis.set_label_coords(-0.17, 0.5)

ax0.vlines(1., -3., 3., lw=0.9)
ax0.set_xlim((-20., 0.1))
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1.)
ax0.plot([0.7, 0.72], [-0.05, 0.03], **kwargs)
ax0.plot([0.73, 0.75], [-0.05, 0.03], **kwargs)

# Real part > 0
ax1 = pl.axes([pos.x0 + 0.18,
               pos.y0,
               0.35 * (pos.x1 - pos.x0),
               pos.y1 - pos.y0])
ax1.spines['right'].set_color('none')
ax1.spines['left'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.set_yticks([])
ax1.plot(np.real(EV[0]),
         np.imag(EV[0]), '.')
critical_eval = EV[0][np.argsort(np.real(EV[0]))[-1]]
ax1.plot(np.real(critical_eval),
         np.imag(critical_eval), '.', color=myred)

ax1.vlines(1., -3., 3., lw=0.7)
ax1.set_xlim((0., 1.))

"""
Panel F: Projection of critical eigenvector onto network
"""
ax = axes['F']
pos = ax.get_position()
divider = make_axes_locatable(ax)
ax_cb = pl.axes([pos.x1,
                 pos.y0,
                 0.02,
                 pos.y1 - pos.y0])

ax_cb.set_frame_on(False)
ax_cb.set_xticks([])
ax_cb.set_yticks([])

critical_eigenvector = np.real(EV[1][:, np.argsort(np.real(EV[0]))[-1]])
r = vector_to_dict(critical_eigenvector, area_list, M.structure)

ev_matrix = np.zeros((8, 32))
for i, area in enumerate(area_list):
    vm = create_vector_mask(M.structure, areas=[area])
    r = critical_eigenvector[vm]
    if area == 'TH':
        r = np.insert(r, 2, np.zeros(2))
    ev_matrix[:, i] = r

ind = [list(area_list).index(area) for area in hierarchical_areas[::-1]]

im = ax.pcolormesh(np.abs(ev_matrix[::-1][:, ind]), cmap=pl.get_cmap('inferno'),
                   norm=LogNorm(vmin=1e-3, vmax=1e0))

area_string_F = area_list[ind][0]
for area in area_list[ind][1:]:
    area_string_F += ' '
    area_string_F += area

ax.set_xlabel('Area')
ax.set_xticks([])
ax.set_yticklabels(population_labels[::-1])
ax.set_yticks(np.arange(8.) + 0.5)
cb = pl.colorbar(im, ax=ax_cb, fraction=1.)
cb.set_ticks([0.001, 1.])
cb.set_ticklabels([r'$0.001$', r'$1$'])
cb.ax.tick_params(labelsize=8, length=0, rotation=0)
ax_cb.text(1.2, 0.8, 'critical eigenvector',
           rotation=90, transform=ax.transAxes)


"""
Create 100 surrogate matrices by shuffling the cross-correlation peak
matrix and measure the violations of the temporal hierarchy to judge
the significance of the temporal hierarchy.
"""
print("Surrogate matrices")
surrogate_matrix = copy.deepcopy(cc_matrix_without_MDP)
violation_list = []
np.random.seed(123)
for i in range(1):
    for j in range(len(area_list_without_MDP)):
        ind = np.extract(np.arange(len(area_list_without_MDP)) != i,
                         np.arange(len(area_list_without_MDP)))
        ind = np.arange(j, len(area_list_without_MDP))
        surr = surrogate_matrix[j][ind][np.random.shuffle(ind)]
        surrogate_matrix[j][ind] = surr
        surrogate_matrix[:, j][ind] = -1.*surr
    (surrogate_res, surrogate_hierarchy,
     surrogate_hierarchical_areas,
     sit) = create_hierarchy(surrogate_matrix, area_list_without_MDP)
    violation_list.append(count_violations(hierarchy, surrogate_matrix[sit][:, sit]))

print("Mean violations of surrogates: ", np.mean(violation_list), " +- ", np.std(violation_list))

print(("Violations of hierarchy: ", count_violations(
    hierarchy, cc_matrix[index_transformation][:, index_transformation])))


for label in ['E', 'G', 'G2']:
    axes[label].spines['right'].set_color('none')
    axes[label].spines['left'].set_color('none')
    axes[label].spines['top'].set_color('none')
    axes[label].spines['bottom'].set_color('none')
    axes[label].yaxis.set_ticks_position("none")
    axes[label].xaxis.set_ticks_position("none")
    axes[label].set_xticks([])
    axes[label].set_yticks([])


"""
Plot the colorbar for the surface plots.
"""
ax = axes['G2']

sm = pl.cm.ScalarMappable(cmap=pl.get_cmap('inferno_r'), norm=pl.Normalize(
    vmax=np.min(hierarchy), vmin=np.max(hierarchy)))
sm.set_array([])
cbticks = []
cbar = pl.colorbar(sm, ax=ax, ticks=cbticks, shrink=0.9)
ax.annotate('', xy=(1.3, 0.9), xycoords='axes fraction',
            xytext=(1.3, 0.1), arrowprops=dict(arrowstyle="->", color='k'))
ax.text(1.45, 23., 'Temporal hierarchy', rotation=90)

pl.text(0.02, 0.1, r'\bfseries{}Order of cortical areas', transform=fig.transFigure)
pl.text(0.02, 0.08, ' '.join((r'\textbf{A}:', area_string_A)),
        transform=fig.transFigure, fontsize=7)
pl.text(0.02, 0.06, ' '.join((r'\textbf{C}:', area_string_C)),
        transform=fig.transFigure, fontsize=7)
pl.text(0.02, 0.04, ' '.join((r'\textbf{D}:', area_string_D)),
        transform=fig.transFigure, fontsize=7)
pl.text(0.02, 0.02, ' '.join((r'\textbf{F}:', area_string_F)),
        transform=fig.transFigure, fontsize=7)


"""
Save figure
"""
pl.savefig('Fig7_temporal_hierarchy_mpl.eps')


"""
Merge surface plots
"""
c = pyx.canvas.canvas()

c.insert(pyx.epsfile.epsfile(
    0., 0., "Fig7_temporal_hierarchy_mpl.eps", width=18.))
c.insert(pyx.epsfile.epsfile(
    2., 2.2, "Fig7_surface_plot_lateral.eps", width=5.5))
c.insert(pyx.epsfile.epsfile(
    8., 2.2, "Fig7_surface_plot_medial.eps", width=5.5))

c.writeEPSfile("Fig7_temporal_hierarchy.eps")
