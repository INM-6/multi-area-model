import os
import numpy as np
import pylab as pl
from plotfuncs import create_fig
from plotcolors import myblue, myred
from area_list import area_list, population_labels
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_vector_mask, create_mask
from multiarea_model.multiarea_helpers import matrix_to_dict, area_level_dict
import utils

base_dir = os.getcwd()
cmap = pl.cm.coolwarm
cmap = cmap.from_list('mycmap', [myblue, 'white', myred], N=256)


"""
Figure layout
"""
scale = 1.
width = 5.2  # inches for 1.5 JoN columns
n_horz_panels = 2.
n_vert_panels = 2.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.22, hoffset=0.087, squeeze=0.25)


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
K_prime1 = np.load('iteration_1/K_prime.npy')
K_prime2 = np.load('iteration_2/K_prime.npy')
K_prime3 = np.load('iteration_3/K_prime.npy')
K_prime4 = np.load('iteration_4/K_prime.npy')


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Changes of the indegree matrix")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Iteration 1: {}".format(np.sum(K_prime1 - K_default) / np.sum(K_default)))
print("Iteration 2: {}".format(np.sum(K_prime2 - K_prime1) / np.sum(K_prime1)))
print("Iteration 3: {}".format(np.sum(K_prime3 - K_prime2) / np.sum(K_prime2)))
print("Iteration 4: {}".format(np.sum(K_prime4 - K_prime3) / np.sum(K_prime3)))
print("In total: {}".format(np.sum(K_prime4 - K_default) / np.sum(K_default)))

data = {}
for iteration in [1, 2, 3, 4, 5]:
    data[iteration] = utils.load_iteration(iteration)

"""
Panel A
"""
ax = panel_factory.new_panel(0, 0, 'A', label_position='leftleft')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

mask = create_vector_mask(M_base.structure, pops=['5E', '6E'])
for ii, iteration in enumerate([1, 2, 3, 4, 5]):
    pl.plot(data[iteration]['parameters'],
            np.mean(data[iteration]['results'][:, mask, -1], axis=1), '.-',
            color=str(ii / 6.))


ax.set_yscale('Log')
ax.yaxis.set_minor_locator(pl.NullLocator())
ax.set_yticks(10 ** np.arange(-1., 3., 1.))
ax.yaxis.set_label_coords(-0.13, 0.55)
ax.set_ylabel(r'$\langle \nu_{\{\mathrm{5E,6E}\}} \rangle$')
ax.set_xlabel(r'$\kappa$', labelpad=-0.1)

ax.set_xlim((1., 1.23))


"""
Panel B
"""
ax = panel_factory.new_panel(1, 0, 'B', label_position=-0.25)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('bottom')

K_prime1_dict = matrix_to_dict(K_prime1,
                               area_list, M_base.structure,
                               external=M_base.K_matrix[:, -1])
K_prime1_area_dict = area_level_dict(K_prime1_dict, M_base.N)

dev_areas_matrix = np.zeros((len(area_list), len(area_list)))
for ii, area1 in enumerate(area_list[::-1]):
    for jj, area2 in enumerate(area_list):
        dev_areas_matrix[ii][jj] = ((K_prime1_area_dict[area1][area2]
                                     - M_base.K_areas[area1][area2]) /
                                    sum(K_prime1_area_dict[area1].values()))

clim = max(abs(np.min(dev_areas_matrix)), abs(np.max(dev_areas_matrix)))
im = ax.pcolormesh(dev_areas_matrix, cmap=cmap, vmin=-clim, vmax=clim)
ax.set_xlim((0, 32))
ax.set_ylim((0, 32))


ax.set_xticks([0.5, 3.5, 14.5, 24.5, 28.5])
ax.xaxis.set_major_locator(pl.FixedLocator([0, 1, 4, 9, 24, 31]))
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.xaxis.set_minor_locator(pl.FixedLocator([0.5, 2.5, 6.5, 16.5, 27.5, 31.5]))
ax.set_xticklabels([8, 7, 6, 5, 4, 2], minor=True)
ax.tick_params(axis='x', which='minor', length=0.)
ax.set_xlabel('Arch. type', labelpad=-0.3)


ax.set_yticks([0.5, 3.5, 14.5, 24.5, 28.5])
ax.yaxis.set_major_locator(pl.FixedLocator([0, 1, 4, 9, 24, 31]))
ax.yaxis.set_major_formatter(pl.NullFormatter())
ax.yaxis.set_minor_locator(pl.FixedLocator(
    [0.5, 4.5, 15.5, 25.5, 28.5, 31.5]))
ax.set_yticklabels([2, 4, 5, 6, 7, 8], minor=True)
ax.tick_params(axis='y', which='minor', length=0.)
ax.set_ylabel('Arch. type')


t = pl.FixedLocator([-0.1, -0.05, 0, 0.05, 0.1])
pl.colorbar(im, ticks=t)


"""
Panel C
"""
zoom = []
for a1, area1 in enumerate(['FEF', '46']):
    for p1, pop1 in enumerate(M_base.structure[area1]):
        zoom.append((area1, pop1))

dev_zoom = np.zeros((16, 16))
for ii in range(16):
    area1, pop1 = zoom[ii]
    mask2 = create_mask(M_base.structure, target_areas=[area1], target_pops=[pop1])
    for jj in range(16):
        area2, pop2 = zoom[jj]
        mask = create_mask(M_base.structure, target_areas=[area1],
                           source_areas=[area2],
                           target_pops=[pop1],
                           source_pops=[pop2],
                           external=False)
        dev_zoom[ii][jj] = (K_prime1[mask[:, :-1]] -
                            K_default[mask[:, :-1]]) / np.sum(K_default[mask2[:, :-1]])


ax = panel_factory.new_panel(0, 1, 'C', label_position=-0.25)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')

clim = max(abs(np.min(dev_zoom)), abs(np.max(dev_zoom)))
im = ax.pcolormesh(dev_zoom[::-1], cmap=cmap, vmin=-clim, vmax=clim)

tick_labels = population_labels + population_labels


ax.set_xticklabels(['FEF', '46'])
ax.set_xticks([4, 12])

ax.set_yticklabels(['FEF', '46'][::-1])
ax.set_yticks([4, 12])

ax.add_patch(pl.Rectangle((0, 8), 8, 8, fill=False))
ax.add_patch(pl.Rectangle((0, 0), 8, 8, fill=False))
ax.add_patch(pl.Rectangle((8, 0), 8, 8, fill=False))
ax.add_patch(pl.Rectangle((8, 8), 8, 8, fill=False))


ax.set_xlabel('Source population', labelpad=-0.2)
ax.set_ylabel('Target population', labelpad=-0.5)


t = pl.FixedLocator([-0.1, -0.05, 0, 0.05, 0.1])
pl.colorbar(im, ticks=t)

"""
Panel D
"""

ax = panel_factory.new_panel(1, 1, 'D', label_position=-0.25)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('bottom')

dev = K_prime4 - K_prime1
dev_matrix = np.zeros((32, 8))
for j, pop in enumerate(M_base.structure['V1']):
    for i, area in enumerate(area_list):
        mask = create_mask(M_base.structure, source_areas=[
                               area], source_pops=[pop])
        dev_matrix[i, j] = np.sum(dev[mask[:, :-1]]) / np.sum(K_prime1[mask[:, :-1]])
        mask2 = create_mask(M_base.structure, target_areas=[
                                area], target_pops=[pop], source_areas=[area])

clim = max(abs(np.min(dev_matrix[np.isfinite(dev_matrix)])), abs(
    np.max(dev_matrix[np.isfinite(dev_matrix)])))
dev_matrix = np.transpose(dev_matrix)
dev_matrix_masked = np.ma.masked_where(np.isnan(dev_matrix), dev_matrix)
im = ax.pcolormesh(dev_matrix_masked[::-1], cmap=cmap, vmin=-clim, vmax=clim)


ax.set_xticks([0.5, 3.5, 14.5, 24.5, 28.5])
ax.xaxis.set_major_locator(pl.FixedLocator([0, 1, 4, 9, 24, 31]))
ax.xaxis.set_major_formatter(pl.NullFormatter())
ax.xaxis.set_minor_locator(pl.FixedLocator([0.5, 2.5, 6.5, 16.5, 27.5, 31.5]))
ax.set_xticklabels([8, 7, 6, 5, 4, 2], minor=True)
ax.tick_params(axis='x', which='minor', length=0.)
ax.set_xlabel('Arch. type', labelpad=-0.3)

y_index = list(range(8))
y_index = [a + 0.5 for a in y_index]

ax.set_xlim(0, 32)
ax.set_yticklabels(population_labels)
ax.set_yticks(y_index[::-1])

t = pl.FixedLocator([-0.5, -0.25, 0, 0.25, 0.5])
pl.colorbar(im, ticks=t)


"""
Save figure
"""
path = '.'
pl.savefig('Fig7_stabilization_analysis.eps')
