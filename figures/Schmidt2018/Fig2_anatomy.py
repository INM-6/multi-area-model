import json
import numpy as np
import matplotlib.pyplot as pl
import os
from helpers import area_list, datapath
from plotcolors import myblue, myblue2, myred, myyellow, myred2
from plotfuncs import create_fig
from scipy import stats
colors = [myblue2, myblue,  myyellow, myred2, myred]

NEURON_DENSITIES_AVAILABLE = False

if NEURON_DENSITIES_AVAILABLE:
    with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
        proc = json.load(f)
    neuron_densities = proc['neuronal_densities']
    architecture_completed = proc['architecture_completed']

    categories = {}
    for i in np.arange(0, 9, 1):
        categories[i] = []
    for area in architecture_completed:
        categories[architecture_completed[area]].append(area)

    with open(os.path.join(datapath, 'viscortex_raw_data.json'), 'r') as f:
        raw = json.load(f)
    thicknesses = raw['laminar_thicknesses']
    total_thickness_data = raw['total_thickness_data']

    # calculate average relative layer thicknesses for each area
    frac_of_total = {}
    for area in list(thicknesses.keys()):
        area_dict_total = {}
        for layer in list(thicknesses[area].keys()):
            area_dict_total[layer] = np.array(
                thicknesses[area][layer]) / np.array(thicknesses[area]['total'])
            # if layer thickness is zero, it makes up 0% of the total, even if the
            # total is unknown
            if 0 in np.array(thicknesses[area][layer]):
                if np.isscalar(thicknesses[area][layer]):
                    area_dict_total[layer] = 0
                else:
                    indices = np.where(np.array(
                        thicknesses[area][layer]) == 0)[0]
                    for i in indices:
                        area_dict_total[layer][i] = 0
        frac_of_total[area] = area_dict_total

    total = {}
    for area in list(thicknesses.keys()):
        totals = thicknesses[area]['total']
        if not np.isscalar(totals):
            if sum(np.isfinite(totals)):
                total[area] = np.nansum(totals) / sum(np.isfinite(totals))
            else:
                total[area] = np.nan
        else:
            total[area] = totals

    # Create arrays

    frac1_of_total = np.zeros(len(area_list))
    frac23_of_total = np.zeros(len(area_list))
    frac4_of_total = np.zeros(len(area_list))
    frac5_of_total = np.zeros(len(area_list))
    frac6_of_total = np.zeros(len(area_list))

    for i, area in enumerate(area_list):
        temp = frac_of_total[area]['1']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac1_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac1_of_total[i] = np.nan
        else:
            frac1_of_total[i] = temp
        temp = frac_of_total[area]['23']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac23_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac23_of_total[i] = np.nan
        else:
            frac23_of_total[i] = temp
        temp = frac_of_total[area]['4']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac4_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac4_of_total[i] = np.nan
        else:
            frac4_of_total[i] = temp
        temp = frac_of_total[area]['5']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac5_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac5_of_total[i] = np.nan
        else:
            frac5_of_total[i] = temp
        temp = frac_of_total[area]['6']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac6_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac6_of_total[i] = np.nan
        else:
            frac6_of_total[i] = temp


    total_array = np.zeros(len(area_list))
    for i, area in enumerate(area_list):
        total_array[i] = total[area]

    architecture_array = np.zeros(len(area_list))
    log_density_array = np.zeros(len(area_list))
    for i, area in enumerate(area_list):
        architecture_array[i] = architecture_completed[area]
        log_density_array[i] = np.log10(neuron_densities[area]['overall'])


    # ################################################################################
    scale = 1.0
    width = 7.5
    n_horz_panels = 3.
    n_vert_panels = 1.
    panel_factory = create_fig(1, scale, width, n_horz_panels,
                               n_vert_panels, hoffset=0.06, voffset=0.19, height_sup=.2)

    axes = {}
    axes['A'] = panel_factory.new_panel(0, 0, r'A', label_position=(-0.2, 1.2))
    axes['B'] = panel_factory.new_panel(1, 0, r'B', label_position=(-0.2, 1.2))
    axes['C'] = panel_factory.new_panel(2, 0, r'C', label_position=(-0.2, 1.2))

    labels = ['A', 'B', 'C']
    for label in labels:
        axes[label].spines['right'].set_color('none')
        axes[label].spines['top'].set_color('none')
        axes[label].yaxis.set_ticks_position("left")
        axes[label].xaxis.set_ticks_position("bottom")


    ax = axes['A']
    x = np.arange(1, 9, 1)
    y = np.array([])
    for cat in x:
        y = np.append(y, proc['category_density'][str(cat)]['overall'])

    layers = ['6', '5', '4', '23', '1'][::-1]
    layer_labels = ['6', '5', '4', '2/3', '1'][::-1]

    rho = {'1': np.array([]), '23': np.array([]), '4': np.array(
        []), '5': np.array([]), '6': np.array([])}
    # Define a prototype area for each category, for which we do not have
    # laminar data so that its laminar thicknesses are computed from the
    # regression
    prototype = {'2': 'TH', '4': 'AITd', '5': 'CITd',
                 '6': 'V3A', '7': 'V2', '8': 'V1'}
    for l in layers:
        if l != '1':
            for cat in x:
                if cat not in [1, 3]:
                    rho[l] = np.append(rho[l], proc['category_density'][str(cat)][
                                       l] * proc['laminar_thicknesses'][prototype[str(cat)]][l])
                else:
                    rho[l] = np.append(rho[l], np.nan)
        else:
            rho[l] = np.zeros(8)

    bottom = np.zeros(8)
    for l in layers[:]:
        bottom += rho[l]

    for i, l in enumerate(layers):
        print(l,  rho[l][4], bottom[4])
        bottom -= rho[l]
        print("LAYER", l, bottom)
        ax.bar(x - 0.4, rho[l], bottom=bottom,
               color=colors[i], label='L' + layer_labels[i],
               edgecolor='k')

    ax.set_xlabel('Architectural type', labelpad=0.3)
    ax.set_ylabel(r'Neuron density ($10^4$/mm$^2$)')
    yticklocs = [50000, 100000, 150000, 200000]
    ytickslabels = ['5', '10', '15', '20']
    ax.set_yticks(yticklocs)
    ax.set_yticklabels(ytickslabels)
    ax.set_ylim((0., 200000.))
    ax.set_xlim((-0.5, 9))
    ax.set_xticks(np.arange(2, 9, 1) - 0.4)
    ax.set_xticklabels([r'${}$'.format(i) for i in list(range(2, 9))])

    ax.legend(loc=(0.035, 0.45), edgecolor='k')

    ##################################################
    barbas_array = np.zeros(len(area_list))
    for i, area in enumerate(area_list):
        barbas_array[i] = total_thickness_data[area] / 1000.

    gradient, intercept, r_value, p_value, std_err = stats.linregress(
        log_density_array[np.isfinite(barbas_array)],
        barbas_array[np.isfinite(barbas_array)])

    print('total thicknesses from Barbas lab vs log. densities:')
    print('gradient: ', gradient)
    print('intercept: ', intercept)
    print('r-value: ', r_value)
    print('p-value: ', p_value)

    ax = axes['B']

    ax.plot(log_density_array, barbas_array, '.', ms=6, color='k')
    line = gradient * log_density_array + intercept
    ax.plot(log_density_array, line, '-', linewidth=1.5, color='k')
    ax.set_xlabel('Log neuron density', labelpad=0.3)
    ax.set_ylabel('Total thickness (mm)')
    ax.set_xticks([4.7, 5.0])

    ax.set_yticks(np.arange(1., 3., 0.5))

    ax = axes['C']

    print('fractions of total thickness vs log. densities')
    layers = ['1', '23', '4', '5', '6']
    for i, data in enumerate([frac1_of_total, frac23_of_total,
                              frac4_of_total, frac5_of_total, frac6_of_total]):
        ax.plot(log_density_array, data, '.', c=colors[i], ms=6)
        gradient, intercept, r_value, p_value, std_err = stats.linregress(
            log_density_array[np.isfinite(data)], data[np.isfinite(data)])
        print('r: ', r_value, ', p-value: ', p_value)
        line = gradient * log_density_array + intercept
        ax.plot(log_density_array, line, '-', linewidth=2.0, c=colors[i])

    ax.set_xlabel('Log neuron density', labelpad=0.3)
    ax.set_ylabel('Proportion of \n total thickness')
    ax.set_xlim((4.6, 5.3))
    ax.set_xticks([4.7, 5.0])
    ax.set_yticks(np.arange(0., 0.7, 0.2))

    pl.savefig('Fig2_anatomy.eps', dpi=600)
else:
    print("Figure 2 can currently not be produced because "
          "we cannot publish the underlying raw data.")
