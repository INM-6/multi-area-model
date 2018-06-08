import matplotlib.pyplot as pl
import numpy as np
import json
import pyx
import os
import subprocess

from config import base_path
from helpers import area_list, population_labels, layer_labels
from helpers import datapath, raw_datapath
from scipy import integrate
from matplotlib import rc_file, gridspec
from plotcolors import myred, myblue
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask

NEURON_DENSITIES_AVAILABLE = False

if NEURON_DENSITIES_AVAILABLE:
    """
    Layout
    """
    rc_file('plotstyle.rc')

    nrows = 2.2
    ncols = 2
    width = 6.8504
    panel_wh_ratio = 0.7 * (1. + np.sqrt(5)) / 2.  # golden ratio

    height = width / panel_wh_ratio * float(nrows) / ncols

    print(width, height)
    pl.rcParams['figure.figsize'] = (width, height)

    axes = {}
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.1, right=0.47, top=0.95, bottom=0.75, wspace=0.1, hspace=0.3)
    axes['A'] = pl.subplot(gs1[:1, :1])

    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(left=0.1, right=0.47, top=0.65, bottom=0.1, wspace=0.1, hspace=0.8)
    axes['B'] = pl.subplot(gs2[:1, :1])
    axes['B2'] = pl.subplot(gs2[1:2, :1])
    axes['B3'] = pl.subplot(gs2[2:3, :1])

    gs3 = gridspec.GridSpec(3, 1)
    gs3.update(left=0.6, right=0.95, top=0.65, bottom=0.1, wspace=0.1, hspace=0.8)
    axes['D'] = pl.subplot(gs3[:1, :1])
    axes['D2'] = pl.subplot(gs3[1:2, :1])
    axes['D3'] = pl.subplot(gs3[2:3, :1])

    axes['D'].set_xticks([])
    axes['D'].set_yticks([])

    axes['D2'].set_xticks([])
    axes['D2'].set_yticks([])

    gs4 = gridspec.GridSpec(1, 1)
    gs4.update(left=0.6, right=0.95, top=0.95, bottom=0.75, wspace=0.1, hspace=0.3)

    axes['C'] = pl.subplot(gs4[:1, :1], frameon=False)
    axes['C'].set_yticks([])
    axes['C'].set_xticks([])

    for label in ['A', 'B', 'B2', 'B3']:
        axes[label].spines['right'].set_color('none')
        axes[label].spines['top'].set_color('none')
        axes[label].yaxis.set_ticks_position("left")
        axes[label].xaxis.set_ticks_position("bottom")

    labels = ['A', 'B', 'C', 'D']
    for label in labels:
        if label in ['A', 'C', 'D']:
            label_pos = [-0.2, 1.04]
        else:
            label_pos = [-0.2, 1.1]
        pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
                fontdict={'fontsize': 10, 'weight': 'bold',
                          'horizontalalignment': 'left', 'verticalalignment':
                          'bottom'}, transform=axes[label].transAxes)

    data = np.loadtxt(os.path.join(raw_datapath, 'RData_prepared_logdensities.txt'),
                      skiprows=3, dtype='S10')
    target = np.array(data[:, 1], dtype=str)
    source = np.array(data[:, 2], dtype=str)

    x = np.array(data[:, 7], dtype=np.float)
    TOT = np.array(data[:, 5], dtype=np.float)

    S = np.array(data[:, 3], dtype=np.float)
    I = TOT - S

    with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
        proc = json.load(f)
    with open(os.path.join(datapath, 'viscortex_raw_data.json'), 'r') as f:
        raw = json.load(f)

    thom_distance_data = raw['thom_distance_data']
    thom_distance_data_markov = raw['thom_distance_data_markov']
    hierarchy_markov = raw['hierarchy_markov']
    overlap = raw['overlap']

    neuron_densities = proc['neuronal_densities']
    SLN_Data = proc['SLN_completed']
    SLN_Data_FV91 = proc['SLN_Data_FV91']
    cocomac = proc['cocomac_completed']

    for area in hierarchy_markov:
        if area not in neuron_densities and area != 'PERIRHINAL':
            print(area)
            x = 0.
            y = 0.
            area_key = area
            if area == 'TH/TF':
                area_key = 'TH_TF'
            if area == '8L':
                area_key = '8l'
            for FV91 in overlap['visual'][area_key + '_M132']:
                if 'FVE' in FV91:
                    x += overlap['visual'][area_key + '_M132'][FV91] / 100. * \
                        neuron_densities[str(FV91).split('.')[-1]]['overall']
                    y += overlap['visual'][area_key + '_M132'][FV91] / 100.
            neuron_densities[area] = {'overall': x / y}

    def integrand(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    def probit(x,):
        if isinstance(x, np.ndarray):
            res = [integrate.quad(integrand, -1000., i,
                                  args=(0., 1.))[0] for i in x]
        else:
            res = integrate.quad(integrand, -1000., x, args=(0., 1.))[0]
        return res

    def chi2(x, y, z, n):
        return 1. / (n) * np.sum(((x - y) / z) ** 2)

    """
    Fit of SLN vs. architectural type differences
    """
    SLN_array = np.array(data[:, 10], dtype=np.float)
    densities = np.array(data[:, 7], dtype=np.float)

    # Call R script to perform SLN fit
    try:
        proc = subprocess.Popen(["Rscript",
                                 os.path.join(datapath, 'SLN_logdensities.R'),
                                 base_path],
                                stdout=subprocess.PIPE)
        out = proc.communicate()[0].decode('utf-8')
        R_fit = [float(out.split('\n')[1].split(' ')[1]),
                 float(out.split('\n')[1].split(' ')[3])]
    except OSError:
        print("No R installation, taking hard-coded fit parameters.")
        R_fit = [-0.1516142, -1.5343200]

    print("We currently cannot publish the R code because of "
          "copyright issues, there taking hard-coded fit parameters. "
          "See Schmidt et al. (2018) for a full explanation "
          "of the procedure.")
    R_fit = [-0.1516142, -1.5343200]
    print(R_fit)

    ax = axes['A']
    ax.plot(densities, SLN_array, '.', linewidth=1.5, color=myblue)
    x = np.arange(-2., 2., 0.1)
    ax.plot(x, np.array(
        probit((R_fit[1] * x + R_fit[0]))), '-', linewidth=1.5, color='k')
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(r'$SLN$')
    ax.set_xlabel(r'Log ratio of densities')
    ax.xaxis.set_label_coords(0.5, -0.16)
    ax.set_xticks([-2., -1., 0., 1., 2.])
    ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
    goodness_bb = round(chi2(SLN_array, probit(
        R_fit[1] * densities + R_fit[0]), np.ones(SLN_array.size), SLN_array.size - 2), 4)
    corr_bb = round(np.corrcoef(
        np.array(probit(R_fit[1] * densities + R_fit[0])), SLN_array)[0][1] ** 2, 2)
    print("SLN Fit: R={}, Chi2={}".format(corr_bb, goodness_bb))


    # Target-source relationship

    target_low_SLN_unweighted = np.zeros(6)
    target_medium_SLN_unweighted = np.zeros(6)
    target_high_SLN_unweighted = np.zeros(6)

    for target in SLN_Data_FV91:
        for source in SLN_Data_FV91[target]:
            if cocomac[target][source]['target_pattern']:
                if SLN_Data_FV91[target][source] < 0.35:
                    for layer in range(6):
                        target_low_SLN_unweighted[layer] += int(int(cocomac[target][
                            source]['target_pattern'][layer]) > 0)
                if 0.35 <= SLN_Data_FV91[target][source] <= 0.65:
                    for layer in range(6):
                        target_medium_SLN_unweighted[layer] += int(int(cocomac[target][
                            source]['target_pattern'][layer]) > 0)
                if SLN_Data_FV91[target][source] > 0.65:
                    for layer in range(6):
                        target_high_SLN_unweighted[layer] += int(int(cocomac[target][
                            source]['target_pattern'][layer]) > 0)

    SLN_labels = [r' $\mathbf{SLN < 0.35}$',
                  r'$\mathbf{0.35\leq SLN \leq 0.65}$',
                  r'$\mathbf{SLN > 0.65}$']

    data_list = [target_low_SLN_unweighted,
                 target_medium_SLN_unweighted,
                 target_high_SLN_unweighted]

    for axlabel, label, data in zip(['B', 'B2', 'B3'], SLN_labels, data_list):
        ax = axes[axlabel]
        ax.yaxis.set_ticks_position("none")
        ax.barh(list(6 - np.arange(6)), list(data), edgecolor='none', color=myblue)
        ax.text(0.1, 1.1, label, transform=ax.transAxes)
        ax.set_yticks(6.4 - np.arange(6))
        ax.set_yticklabels(layer_labels)

        ax.set_xlabel('Count', size=10)
        if axlabel == 'B2':
            ax.set_ylabel('Target layer', size=10)
            ax.yaxis.set_label_coords(-0.15, 0.5)

    # Resulting patterns in the connectivity matrix
    M = MultiAreaModel({})

    FF_conns = []
    FB_conns = []
    lateral_conns = []

    for target_area in area_list:
        for source_area in area_list:
            mask = create_mask(M.structure,
                               target_areas=[target_area],
                               source_areas=[source_area],
                               external=False)
            if (np.sum(M.K_matrix[mask]) > 0 and source_area != target_area
                    and source_area in SLN_Data[target_area]):
                m = M.K_matrix[mask] / np.sum(M.K_matrix[mask])
                if m.size == 64:
                    m = m.reshape((8, 8))
                elif m.size == 48:
                    if target_area == 'TH':
                        m = m.reshape((6, 8))
                        m = np.insert(m, 2, np.zeros((2, 8), dtype=float), axis=0)
                    elif source_area == 'TH':
                        m = m.reshape((8, 6))
                        m = np.insert(m, 2, np.zeros((2, 8), dtype=float), axis=1)
                if SLN_Data[target_area][source_area] < 0.35:
                    FB_conns.append(m)
                elif SLN_Data[target_area][source_area] > 0.65:
                    FF_conns.append(m)
                else:
                    lateral_conns.append(m)

    FF_conns = np.array(FF_conns)
    FB_conns = np.array(FB_conns)
    lateral_conns = np.array(lateral_conns)

    data_list = [FB_conns, lateral_conns, FF_conns]
    label_list = ['Feedback', 'lateral', 'Feedforward']

    for axlabel, label, data in zip(['D', 'D2', 'D3'], label_list, data_list):
        ax = axes[axlabel]
        ax.yaxis.set_ticks_position("none")
        ax.xaxis.set_ticks_position("none")
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        if axlabel == 'D3':
            ax.set_xticks([0.025, 0.625, 1.225])
            ax.set_xlim((0., 1.8))
            factor = 0.6
        else:
            ax.set_xticks([0.025, 0.225, 0.425])
            ax.set_xlim((0., 0.6))
            factor = 0.2

        matrix = np.mean(data, axis=0)
        ind = [0, 4, 6]
        for i in range(3):
            ax.barh(np.arange(8), matrix[:, ind][:, i][
                    ::-1], left=i * factor, color=myred, edgecolor='none')

        ax.text(0.1, 1.1, label, transform=ax.transAxes)
        ax.set_yticks(np.arange(8) + 0.3)
        ax.set_yticklabels(population_labels[::-1], size=8.)

        if axlabel == 'D2':
            ax.set_ylabel('Target population', size=10)
        if axlabel == 'D3':
            ax.set_xlabel('Source population', size=10)

        ax.set_xticklabels(['2/3E', '5E', '6E'], size=8.)

    pl.savefig('Fig5_cc_laminar_pattern_mpl.eps')

    """
    Merge with syn_illustration figure
    """
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(
        0., 0., "Fig5_cc_laminar_pattern_mpl.eps", width=17.3))
    c.insert(pyx.epsfile.epsfile(
        9.3, 12., "Fig5_syn_illustration.eps", width=7.))
    c.writeEPSfile("Fig5_cc_laminar_pattern.eps")

else:
    print("Figure 5 can currently not be produced because "
          "we cannot publish the underlying raw data.")
