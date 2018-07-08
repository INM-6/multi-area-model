from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import os
import numpy as np
import pylab as pl
import json
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from plotcolors import myblue, myblue2, myred2, myred
from area_list import area_list, population_labels
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask
from multiarea_model.multiarea_helpers import matrix_to_dict, area_level_dict

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

cmap = plt.cm.coolwarm
cmap = cmap.from_list(
    'mycmap', [myblue, myblue2, 'white', myred2, myred], N=256)

base_dir = os.getcwd()


"""
Figure layout
"""
scale = 1.

# resolution of figures in dpi
# does not influence eps output
plt.rcParams['figure.dpi'] = 300

# font
plt.rcParams['font.size'] = scale * 8
plt.rcParams['legend.fontsize'] = scale * 8
plt.rcParams['font.family'] = "sans-serif"

plt.rcParams['lines.linewidth'] = scale * 1.0

# size of markers (points in point plots)
plt.rcParams['lines.markersize'] = scale * 3.0
plt.rcParams['patch.linewidth'] = scale * 1.0
plt.rcParams['axes.linewidth'] = scale * 1.0     # edge linewidth

# ticks distances
plt.rcParams['xtick.major.size'] = scale * 4      # major tick size in points
plt.rcParams['xtick.minor.size'] = scale * 2      # minor tick size in points
plt.rcParams['lines.markeredgewidth'] = scale * 0.5  # line width of ticks
plt.rcParams['grid.linewidth'] = scale * 0.5
# distance to major tick label in points
plt.rcParams['xtick.major.pad'] = scale * 4
# distance to the minor tick label in points
plt.rcParams['xtick.minor.pad'] = scale * 4
plt.rcParams['ytick.major.size'] = scale * 4      # major tick size in points
plt.rcParams['ytick.minor.size'] = scale * 2      # minor tick size in points
# distance to major tick label in points
plt.rcParams['ytick.major.pad'] = scale * 4
# distance to the minor tick label in points
plt.rcParams['ytick.minor.pad'] = scale * 4

# ticks textsize
plt.rcParams['ytick.labelsize'] = scale * 8
plt.rcParams['xtick.labelsize'] = scale * 8

# use latex to generate the labels in plots
# not needed anymore in newer versions
# using this, font detection fails on adobe illustrator 2010-07-20
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

plt.rcParams['ps.useafm'] = False   # use of afm fonts, results in small files
# Output Type 3 (Type3) or Type 42 (TrueType)
plt.rcParams['ps.fonttype'] = 3


nrows = 2.
ncols = 2.
width = 4.8  # inches for 1.5 JoN columns
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio
height = width / panel_wh_ratio * float(nrows) / ncols
print(width, height)
plt.rcParams['figure.figsize'] = (width, height)

fig = plt.figure()

gs1 = gridspec.GridSpec(2, 2)
gs1.update(left=0.1, right=0.94, top=0.8, bottom=0.1, hspace=0.6, wspace=0.4)
axes = {}
axes['A'] = plt.subplot(gs1[:1, :1])
axes['B'] = plt.subplot(gs1[:1, 1:2])
axes['C'] = plt.subplot(gs1[1:2, :1])
axes['D'] = plt.subplot(gs1[1:2, 1:2])


for label in ['C', 'D']:
    label_pos = [-0.3, 1.01]
    if label == 'C':
        label_pos = [-0.25, 1.01]
    plt.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
             fontdict={'fontsize': scale * 10, 'weight': 'bold',
                       'horizontalalignment': 'left', 'verticalalignment': 'bottom'},
             transform=axes[label].transAxes)


"""
Load data
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

K_default = M_base.K_matrix[:, :-1]
K_prime4 = np.load('iteration_4/K_prime.npy')

dev = K_prime4 - K_default

datapath = '../../multiarea_model/data_multiarea'
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)

SLN_Data = proc['SLN_Data_FV91']
SLN_completed = proc['SLN_completed']
FLN_completed = proc['FLN_completed']
architecture_completed = proc['architecture_completed']

architectural_types = []
for area in area_list:
    architectural_types.append(architecture_completed[area])
architectural_types = np.array(architectural_types)

"""
Panel A
"""
ax = axes['A']
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')

zoom = []
for a1, area1 in enumerate(['V4', 'CITv']):
    for p1, pop1 in enumerate(M_base.structure[area1]):
        zoom.append((area1, pop1))

dev = K_prime4 - K_default
dev_zoom = np.zeros((16, 16))
for ii in range(16):
    area1, pop1 = zoom[ii]
    mask2 = create_mask(M_base.structure,
                        target_areas=[area1],
                        target_pops=[pop1])[:, :-1]
    for jj in range(16):
        area2, pop2 = zoom[jj]
        mask = create_mask(M_base.structure,
                           target_areas=[area1],
                           source_areas=[area2],
                           target_pops=[pop1],
                           source_pops=[pop2])[:, :-1]
        dev_zoom[ii][jj] = dev[mask] / np.sum(K_default[mask2])
clim = max(abs(np.min(dev_zoom)), abs(np.max(dev_zoom)))
clim = 0.06
im = ax.pcolormesh(dev_zoom[::-1], cmap=cmap, vmin=-clim, vmax=clim)

tick_labels = population_labels + population_labels

ax.set_xticks([4, 12])
ax.set_xticklabels(['V4', 'CITv'])
ax.set_xlabel('Source population')

ax.set_yticks([4, 12])
ax.set_yticklabels(['CITv', 'V4'])
ax.set_ylabel('Target population', labelpad=-0.1)

t = FixedLocator([-0.06, -0.03, 0, 0.03, 0.06])
plt.colorbar(im, ticks=t, ax=ax)

ax.add_patch(Rectangle((0, 8), 8, 8, fill=False))
ax.add_patch(Rectangle((0, 0), 8, 8, fill=False))
ax.add_patch(Rectangle((8, 0), 8, 8, fill=False))
ax.add_patch(Rectangle((8, 8), 8, 8, fill=False))

pos = ax.get_position()
# Top panel
ax = pl.axes((pos.x0, pos.y1 + 0.07, pos.x1 - pos.x0, 0.1))
label_pos = [-0.35, 0.95]
plt.text(label_pos[0], label_pos[1], r'\bfseries{}' + 'A',
         fontdict={'fontsize': scale * 10,
                   'weight': 'bold',
                   'horizontalalignment': 'left',
                   'verticalalignment': 'bottom'},
         transform=ax.transAxes)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('none')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

dev = K_prime4 - K_default

dev_matrix = np.zeros((32, 8))
for j, pop in enumerate(M_base.structure['V1']):
    for i, area in enumerate(area_list):
        mask = create_mask(M_base.structure,
                           source_areas=[area],
                           source_pops=[pop],
                           target_areas=[area])[:, :-1]
        dev_matrix[i, j] = np.sum(dev[mask]) / np.sum(K_default[mask])

change = []
for ii in range(8):
    ind = np.isfinite(dev_matrix[:, ii])
    change.append(np.mean(dev_matrix[:, ii][ind]))

ax.bar(np.arange(8) + 0.1, change, width=0.8, color=myblue, linewidth=0.)
ax.set_xticklabels(population_labels, rotation=45)
ax.set_xticks(np.arange(8) + 0.45)
ax.set_yticks([0., -0.1])


# Correlation between IA changes
IA_conns = []
IA_conns_relative = np.array([])
for ii, area in enumerate(area_list):
    mask = create_mask(M_base.structure, target_areas=[area])[:, :-1]
    n = K_default[mask]
    mask = create_mask(M_base.structure,
                       target_areas=[area],
                       source_areas=[area])[:, :-1]
    m = dev[mask]
    m_rel = dev[mask] / K_default[mask]
    if m.size == 64:
        n = n.reshape((8, 254))
        m = m.reshape((8, 8))
        for ii in range(8):
            m[ii] /= np.sum(n, axis=1)[ii]
    elif m.size == 36:
        m = m.reshape((6, 6))
        m = np.insert(m, 2, np.zeros((2, 6), dtype=float), axis=0)
        m = np.insert(m, 2, np.zeros((2, 8), dtype=float), axis=1)

    IA_conns.append(m.flatten())
    IA_conns_relative = np.append(IA_conns_relative, m_rel.flatten())

IA_conns = np.array(IA_conns)
IA_conns_relative = np.array(IA_conns_relative)

IA_conns_masked = np.ma.masked_where(np.isnan(IA_conns), IA_conns)
IA_conns_relative_masked = np.ma.masked_where(
    np.isnan(IA_conns_relative), IA_conns_relative)
IA_conns_relative_masked = np.ma.masked_where(
    np.isinf(IA_conns_relative_masked), IA_conns_relative_masked)

print("Average relative modification of intra-areal connections ", end=' ')
np.mean(abs(IA_conns_relative_masked))

D = np.clip(pdist(IA_conns, metric='correlation'), 0, 2)
correlation_matrix = 1. - squareform(D)


"""
Panel B
"""
ax = axes['B']
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')

# Cluster matrix
Z = sch.linkage(D, metric='correlation', method='complete')
clusters = sch.fcluster(Z, 0.7, criterion='distance')

index = np.argsort(clusters)

clim = max(abs(np.min(correlation_matrix)), abs(np.max(correlation_matrix)))
im = ax.pcolormesh(correlation_matrix[index][:, index][
                   ::-1], cmap=cmap, vmin=-clim, vmax=clim)
t = FixedLocator([-1., -0.5, 0, 0.5, 1.])

ax.text(41., 21.6, r'$r_{\mathrm{Pearson}}$', rotation=90)

ax.set_xlim((0, 32))
ax.set_ylim((0, 32))
ax.set_xlabel('Cluster index')
ax.set_ylabel('Cluster index')

xticks = []
yticks = []
for ii in range(1, np.max(clusters) + 1):
    xticks.append(np.median(np.where(np.sort(clusters) == ii)) + .5)
    yticks.append(31.5 - np.median(np.where(np.sort(clusters) == ii)))

ax.set_yticks(yticks)
ax.set_xticks(xticks)
ax.set_xticklabels(np.unique(clusters))
ax.set_yticklabels(np.unique(clusters))


cb = plt.colorbar(im, ticks=t, ax=ax)
pos = ax.get_position()
ax = pl.axes((pos.x0, pos.y1 + 0.07, pos.x1 - pos.x0, 0.1))
label_pos = [-0.35, 0.95]
plt.text(label_pos[0], label_pos[1], r'\bfseries{}' + 'B',
         fontdict={'fontsize': scale * 10,
                   'weight': 'bold',
                   'horizontalalignment': 'left',
                   'verticalalignment': 'bottom'},
         transform=ax.transAxes)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('none')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


m = 0.
colors = [myblue, myred, myblue2, myred2, 'k', '0.5', '0.3', '0.7']

for ii in range(8):
    c = ii + 1
    index = np.where(clusters == c)[0]
    ax.bar(np.arange(m, m + len(index)),
           architectural_types[index],
           width=1., linewidth=0., color=colors[ii])
    m += len(index)

ax.set_xlim((0, 32))
ax.set_xticks(xticks)
ax.set_xticklabels(np.unique(clusters))
ax.set_ylabel('Arch. type')
ax.set_yticks([2, 4, 6, 8])


"""
Panel C
FLN Values
"""
num = M_base.N

K_prime4_dict = matrix_to_dict(K_prime4,
                               area_list, M_base.structure,
                               external=M_base.K_matrix[:, -1])
K_prime4_area_dict = area_level_dict(K_prime4_dict, M_base.N)

K_default_areas = M_base.K_areas

FLN_default = []
FLN_mod = []
FLN_pairs = []
for area in area_list:
    K_tot_default = sum(K_default_areas[area].values())
    K_tot_mod = sum(K_prime4_area_dict[area].values())
    for area2 in area_list:
        if area != area2 and area2 in FLN_completed[area]:
            FLN_pairs.append(area + '-' + area2)
            FLN_default.append(K_default_areas[area][area2] / K_tot_default)
            FLN_mod.append(K_prime4_area_dict[area][area2] / K_tot_mod)

ax = axes['C']
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

ax.set_xlabel(r'$FLN^{(0)}$')
ax.set_ylabel(r'$FLN^{(4)}$', labelpad=-0.1)

FLN_default = np.array(FLN_default)
FLN_mod = np.array(FLN_mod)
FLN_pairs = np.array(FLN_pairs)

ax.plot(np.arange(1e-4, 1., .001), np.arange(1e-4, 1., .001), '-', color='0.5')
ax.plot(FLN_default, FLN_mod, '.', color='k', label='default')
red_pairs = np.where(np.logical_or(
    FLN_pairs == '46-FEF', FLN_pairs == 'FEF-46'))
ax.plot(FLN_default[red_pairs], FLN_mod[red_pairs],
        '.', color=myred, label='default')
ax.set_yscale('Log')
ax.set_xscale('Log')

corr = np.corrcoef(np.log10(FLN_default[FLN_mod != 0]), np.log10(
    FLN_mod[FLN_mod != 0]))[0][1]
print("Correlation of log10 FLN values: ", corr)
corr = np.corrcoef(FLN_default[FLN_mod != 0], FLN_mod[FLN_mod != 0])[0][1]
print("Correlation of FLN values: ", corr)

ax.set_ylim((1e-3, 1.))
ax.set_xlim((1e-3, 1.))
dev = np.log(FLN_default) - np.log(FLN_mod)
dev = dev[np.where(np.logical_and(np.logical_not(
    np.isnan(dev)), np.logical_not(np.isinf(dev))))]

print("Deviation of FLN pre post")
print(np.mean(abs(dev)))


"""
Panel D
SLN values
"""

SLN_default = []
SLN_default_measured = []

SLN_mod = []
SLN_mod_measured = []

log_ratio_densities = []
log_ratio_densities_measured = []
SLN_values = []
pairs = []

for area in K_prime4_dict:
    for area2 in K_prime4_dict:
        if area != area2:
            supra_mod = 0.
            infra_mod = 0.
            supra_default = 0.
            infra_default = 0.
            for pop in K_prime4_dict[area]:
                for pop2 in K_prime4_dict[area][pop][area2]:
                    if pop2 in ['23E']:
                        supra_mod += K_prime4_dict[area][pop][area2][pop2] * \
                            num[area][pop]
                        supra_default += M_base.K[area][
                            pop][area2][pop2] * num[area][pop]
                    elif pop2 in ['5E', '6E']:
                        infra_mod += K_prime4_dict[area][pop][area2][pop2] * \
                            num[area][pop]
                        infra_default += M_base.K[area][
                            pop][area2][pop2] * num[area][pop]
            if (infra_default + supra_default) > 0. and (infra_mod + supra_mod) > 0.:
                pairs.append(area + ' ' + area2)
                SLN_default.append(
                    supra_default / (infra_default + supra_default))
                SLN_mod.append(supra_mod / (infra_mod + supra_mod))

                try:
                    SLN_Data[area][area2]
                    SLN_default_measured.append(
                        supra_default / (infra_default + supra_default))
                    SLN_mod_measured.append(
                        supra_mod / (infra_mod + supra_mod))
                except KeyError:
                    pass

ax = axes['D']
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("none")

diff = np.array(SLN_mod) - np.array(SLN_default)
print("Total diff: ", np.mean(diff), np.std(diff))
hist, bins = np.histogram(diff, bins=50)

ax.bar(bins[:-1], hist, width=np.diff(bins)[0], color='k', linewidth=0)

diff_measured = np.array(SLN_mod_measured) - np.array(SLN_default_measured)
hist, bins = np.histogram(diff_measured, bins=bins)


ax.legend()
ax.set_xlabel('$\delta SLN$')
ax.set_ylabel('Frequency')
ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
ax.set_yticks([0, 20, 100])


"""
Save figure
"""
plt.savefig('Fig8_conn_analysis.eps')
