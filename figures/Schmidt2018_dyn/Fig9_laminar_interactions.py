import json
import numpy as np
import os
import pandas as pd
import pyx

from collections import Counter
from helpers import original_data_path
from helpers import structural_gradient
from multiarea_model.multiarea_model import MultiAreaModel
from plotcolors import myblue, myred, mypurple, myred2

from matplotlib import gridspec
import matplotlib.pyplot as pl
from matplotlib import rc_file
rc_file('plotstyle.rc')


"""
Figure layout
"""

nrows = 2
ncols = 3
width = 7.0866
panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

height = width / panel_wh_ratio * float(nrows) / ncols
height = 4.5
print((width, height))
pl.rcParams['figure.figsize'] = (width, height)


"""
Load data
"""
datapath = '../../multiarea_model/data_multiarea'
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)
arch_types = proc['architecture_completed']

conn_params = {'g': -16.,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
               'K_stable': '../SchueckerSchmidt2017/K_prime_original.npy'}
network_params = {'N_scaling': 1.,
                  'K_scaling': 1.,
                  'connection_params': conn_params}
M = MultiAreaModel(network_params)


LOAD_ORIGINAL_DATA = True

if LOAD_ORIGINAL_DATA:
    label = '99c0024eacc275d13f719afd59357f7d12f02b77'
    data_path = original_data_path
else:
    from network_simulations import init_models
    from config import data_path
    models = init_models('Fig7')
    label = models[0].simulation.label

gc = {}
for area in M.area_list:
    gc[area] = {}
    for pop in M.structure[area]:
        fn = os.path.join(data_path,
                          label,
                          'Analysis',
                          'granger_causality',
                          'granger_causality_{}_{}.json'.format(area, pop))
        with open(fn, 'r') as f:
            gc[area][pop] = json.load(f)

with open('Fig9_{}_significant_channels.json'.format(label), 'r') as f:
    significant_channels = json.load(f)
for typ in significant_channels:
    significant_channels[typ] = [tuple(pair) for pair in significant_channels[typ]]

"""
Bottom row
"""
gs1 = gridspec.GridSpec(2, 3)
gs1.update(left=0.1, right=0.95, top=0.95, wspace=0.4, bottom=0.3)

for i, ax_label in enumerate(['A', 'B']):
    ax = pl.subplot(gs1[i:i + 1, :])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0., 1.1, r'\bfseries{}' + ax_label, transform=ax.transAxes)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(left=0.1, right=0.95, top=0.255, wspace=0.4, bottom=0.1)

"""
Panel C: Percentage of significant connections for each type of connection
"""
ax = pl.subplot(gs1[:1, :1])
ax.text(0., 1.2, r'\bfseries{} C', transform=ax.transAxes)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")

# determine the proportion of significant pairs to total number of connected pairs
prop = {typ: {'total': 0., 'significant': 0.}
        for typ in ['HL', 'LH', 'HZ', 'same-area']}
for target_area in gc:
    for target_pop in gc[target_area]:
        for source_area in gc[target_area][target_pop]:
            grad = structural_gradient(target_area, source_area, arch_types)
            s_total = 0
            s_sign = 0
            for source_pop in gc[target_area][target_pop][source_area]:
                s_total += 1
                if gc[target_area][target_pop][source_area][source_pop][1] < 0.05:
                    s_sign += 1
            prop[grad]['total'] += s_total
            prop[grad]['significant'] += s_sign

colors = ['0.1', '0.1', '0.1', mypurple]
for i, typ in enumerate(['HL', 'HZ', 'HL', 'same-area']):
    ax.bar([(i + 1) / 5.], [prop[typ]['significant'] / prop[typ]['total']],
           width=0.2,
           color=colors[i])

s_total_overall = 0
s_sign_overall = 0
for typ in ['HL', 'LH', 'HZ', 'same-area']:
    s_total_overall += prop[typ]['total']
    s_sign_overall += prop[typ]['significant']

ax.bar([0.], [s_sign_overall / s_total_overall],
       width=0.2,
       color='k')
ax.set_yticks([0., 0.1, 0.2, 0.3])
ax.set_yticklabels([0, 10, 20, 30])
ax.set_ylabel('\% significant \n connections')

ax.set_xticks([0., 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels([r'$\Sigma$', 'HL', 'HZ', 'LH', 'local'],
                   rotation=0)

"""
Panel D: Difference between excitatory and inhibitory connections
"""
ax = pl.subplot(gs1[:, 1:2])
ax.text(0., 1.2, r'\bfseries{} D', transform=ax.transAxes)
ax.text(0.1, 1., r'$\rightarrow$ E', transform=ax.transAxes)
ax.text(0.7, 1., r'$\rightarrow$ I', transform=ax.transAxes)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")

balance_EI = {}
NI_overall = 0
NE_overall = 0
for i, typ in enumerate(['HL', 'HZ', 'HL']):
    C = Counter(significant_channels[typ])
    NI, NE = 0, 0
    for channel in C:
        if 'I' in channel[1]:
            NI += 1
            NI_overall += 1
        elif 'E' in channel[1]:
            NE += 1
            NE_overall += 1
    balance_EI[typ] = {'E': float(NE) / (NE + NI),
                       'I': float(NI) / (NE + NI)}

    ax.bar(np.array([0, 1]) + (i + 1) / 5., [balance_EI[typ]['E'], balance_EI[typ]['I']],
           width=0.2,
           color=[myblue, myred],
           edgecolor='1.')
ax.bar(np.array([0, 1]), [float(NE_overall) / (NE_overall + NI_overall),
                          float(NI_overall) / (NE_overall + NI_overall)],
       width=0.2,
       color=[myblue, myred])
ax.set_xticks([-0.05, 0.18, 0.41, 0.65, 0.95, 1.18, 1.41, 1.65])
ax.set_xticklabels([r'$\Sigma$', 'HL', 'HZ', 'LH',
                    r'$\Sigma$', 'HL', 'HZ', 'LH'],
                   rotation=0)
ax.set_ylabel('Relative proportion')

"""
GC vs. connection strength
"""
ax = pl.subplot(gs1[:, 2:])
ax.text(0., 1.2, r'\bfseries{} E', transform=ax.transAxes)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("none")

dat = pd.DataFrame(columns=['target', 'source', 'K', 'GC', 'p'])
dat_local = pd.DataFrame(columns=['target', 'source', 'K', 'GC', 'p'])
for target in M.area_list:
    for target_pop in M.structure[target]:
        total_indegrees = 0
        for source in gc[target][target_pop]:
            for source_pop in gc[target][target_pop][source]:
                total_indegrees += M.K[target][target_pop][source][source_pop]
        for source in gc[target][target_pop]:
            for source_pop in gc[target][target_pop][source]:
                y = [['-'.join((target, target_pop)), '-'.join((source, source_pop)),
                      M.K[target][target_pop][source][source_pop] /
                      total_indegrees,
                      gc[target][target_pop][source][source_pop][0],
                      gc[target][target_pop][source][source_pop][1]]]
                x = pd.DataFrame(data=y, columns=[
                                 'target', 'source', 'K', 'GC', 'p'])
                if source != target:
                    dat = dat.append(x)
                else:
                    dat_local = dat_local.append(x)

c_sign = dat.p < 0.05
c_sign_local = dat_local.p < 0.05

dx = np.arange(-5., 0., 0.1)
vals, bins = np.histogram(dat[c_sign].K, bins=10**dx)
vals_total, bins_total = np.histogram(dat.K, bins=10**dx)

vals_local, bins_local = np.histogram(dat_local[c_sign_local].K, bins=10**dx)
vals_local_total, bins_local_total = np.histogram(dat_local.K, bins=10**dx)

ax.bar(bins_total[:-1], vals_total, width=np.diff(bins),
       color='0.7', edgecolor='none')
ax.bar(bins_local_total[:-1], vals_local_total,
       width=np.diff(bins), color=myred2, edgecolor='none')

ax.bar(bins_local[:-1], vals_local, width=np.diff(bins),
       color=mypurple, edgecolor='none')
ax.bar(bins[:-1], vals, width=np.diff(bins), color='0.1', edgecolor='none')
ax.plot(np.mean(dat.K), 250., '^', ms=8, color='0.7', markeredgecolor='0.7')
ax.plot(np.mean(dat[c_sign].K), 250.,  '^',
        ms=8, color='0.1', markeredgecolor='0.1')
ax.plot(np.mean(dat_local.K), 250.,  '^', ms=8,
        color=myred2, markeredgecolor=myred2)
ax.plot(np.mean(dat_local[c_sign_local].K), 250., '^',
        ms=8, color=mypurple, markeredgecolor=mypurple)

ax.set_xscale('Log')
ax.set_xlabel('Relative indegree')
ax.set_ylabel('Count')


"""
Save figure
"""
pl.savefig('Fig9_laminar_interactions_mpl.eps')


"""
Merge figure
"""
c = pyx.canvas.canvas()

c.insert(pyx.epsfile.epsfile(
    0., 0., "Fig9_laminar_interactions_mpl.eps", width=17.9))

c.insert(pyx.epsfile.epsfile(
    1.1, 7.5, "Fig9_{}_HL_interactions.eps".format(label), width=4.6))
c.insert(pyx.epsfile.epsfile(
    6.3, 7.5, "Fig9_{}_HZ_interactions.eps".format(label), width=4.6))
c.insert(pyx.epsfile.epsfile(
    12.3, 7.5, "Fig9_{}_LH_interactions.eps".format(label), width=4.6))

c.insert(pyx.epsfile.epsfile(
    1.1, 3.4, "Fig9_{}_HL_paths.eps".format(label), width=4.6))
c.insert(pyx.epsfile.epsfile(
    6.3, 3.4, "Fig9_{}_HZ_paths.eps".format(label), width=4.6))
c.insert(pyx.epsfile.epsfile(
    12.3, 3.4, "Fig9_{}_LH_paths.eps".format(label), width=4.6))

c.writeEPSfile("Fig9_laminar_interactions.eps")
