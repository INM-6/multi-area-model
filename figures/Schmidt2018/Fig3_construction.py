import json
import matplotlib.pyplot as p
import numpy as np
import os
import pyx

from collections import OrderedDict
from helpers import area_list, datapath, population_list
from multiarea_model import MultiAreaModel
from plotfuncs import create_fig

"""
Loading and processing of data
"""
M = MultiAreaModel({})
with open(os.path.join(datapath, 'viscortex_processed_data.json'), 'r') as f:
    proc = json.load(f)
density = proc['neuronal_densities']

if density is None:
    print("Since we cannot publish the underlying density data, we"
          "here hard-code the list of areas sorted by their overall"
          "density.")
    areas_sorted_density = ['V1', 'V2', 'V3', 'VP', 'V4', 'MT', 'V4t',
                            'PITv', 'PITd', 'VOT', 'V3A', 'LIP', 'MIP', 'MDP', 'PO', 'PIP',
                            'MSTl', 'VIP', 'MSTd', 'DP', 'TF', 'FEF', 'CITv', 'CITd', 'AITv',
                            'AITd', 'STPa', 'STPp', 'FST', '46', '7a', 'TH']
else:
    ordered_density = OrderedDict(
        sorted(list(density.items()), key=lambda t: t[1]['overall'], reverse=True))
    areas_sorted_density = list(ordered_density.keys())

average_indegree = {}
for area in areas_sorted_density:
    s = 0
    for pop in population_list:
        for source_area in list(area_list):
            for source_pop in population_list:
                s += M.synapses[area][pop][source_area][source_pop]
        s += M.synapses[area][pop]['external']['external']
    average_indegree[area] = s / M.N[area]['total']

indegrees = []
num_list = []
for area in areas_sorted_density:
    indegrees.append(average_indegree[area])
    num_list.append(M.N[area]['total'])


"""
Layout
"""
scale = 1.0
width = 3.31
n_horz_panels = 1.
n_vert_panels = 2.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.2, hoffset=0.15)

"""
Plotting
"""
ax = panel_factory.new_panel(0, 0, 'A', label_position=-0.2)
ax.set_frame_on(False)
ax.yaxis.set_ticks_position("none")
ax.xaxis.set_ticks_position("none")
ax.set_xticks([])
ax.set_yticks([])


ax = panel_factory.new_panel(0, 1, 'B', label_position=-0.2)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

ax.bar(np.arange(32) + 0.2, indegrees,
       width=0.8, edgecolor='none', color='0.33')
ax.plot([0., 32.], [np.average(indegrees, weights=num_list),
                    np.average(indegrees, weights=num_list)], '--', color='k')
ax.set_xlabel(r'High neuron density ' +
              r'$\rightarrow$' + r' low neuron density')
ax.set_xticks([])
ax.set_ylabel(r'Average indegree $(\times 10^3)$')
ax.set_yticks([5000., 10000., 15000., 20000.])
ax.set_yticklabels([r'5', r'10', r'15', r'20'])

print(("Average indegree across all areas: {}".format(np.average(indegrees,
                                                                 weights=num_list))))

p.savefig('Fig3_construction_mpl.eps')


"""
Merge with syntypes figure
"""
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(
    0.4, 0., "Fig3_construction_mpl.eps", width=8.3))
c.insert(pyx.epsfile.epsfile(1.4, 5.5, "Fig3_syntypes.eps", width=6.3))
c.writeEPSfile("Fig3_construction.eps")
