from multiarea_model import MultiAreaModel
from multiarea_model import stabilize
from multiarea_model.multiarea_helpers import create_vector_mask
import os
import copy
import numpy as np
import pylab as pl
from plotcolors import myblue, myred
from area_list import area_list, population_labels
from plotfuncs import create_fig
import utils

base_dir = os.getcwd()
cmap = pl.cm.rainbow
cmap = cmap.from_list('mycmap', ['white', myred], N=256)
cmap2 = cmap.from_list('mycmap', [myblue, 'white', myred], N=256)

"""
Figure layout
"""
scale = 1.
width = 4.8
n_horz_panels = 2.
n_vert_panels = 2.
panel_factory = create_fig(
    1, scale, width, n_horz_panels, n_vert_panels, voffset=0.22, hoffset=0.096, squeeze=0.23)


"""
Create instance of MultiAreaModel
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

c_target = copy.deepcopy(conn_params)
c_target.update({'fac_nu_ext_5E': 1.2,
                'fac_nu_ext_6E': 10/3.*1.2-7/3.})
network_params_target = {'connection_params': c_target,
                         'input_params': input_params}
M_target = MultiAreaModel(network_params_target, theory=True,
                          theory_spec=theory_params)

"""
Panel A
"""
ax = panel_factory.new_panel(0, 0, 'A', label_position='leftleft')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

# First iteration
data1 = utils.load_iteration(1)
(par_transition, r_low, r_high,
 minima_low, minima_high) = utils.determine_velocity_minima(time, data1)

unstable_low1 = r_low[:, minima_low[1]]
c = copy.deepcopy(conn_params)
c.update({'fac_nu_ext_5E': par_transition,
          'fac_nu_ext_6E': 10/3.*par_transition-7/3.})
network_params = {'connection_params': c,
                  'input_params': input_params}
MAM = MultiAreaModel(network_params, theory=True,
                     theory_spec=theory_params)

S_vector, S, T_vector, T, M = stabilize.S_T(MAM.theory, unstable_low1)
lambda_ev1, u1, v1 = stabilize.eigen_decomp_M(M)

ax.plot(np.real(lambda_ev1), np.imag(lambda_ev1), 's',
        color=myblue, markeredgecolor='none')

# # Second iteration
data2 = utils.load_iteration(2)
(par_transition, r_low, r_high,
 minima_low, minima_high) = utils.determine_velocity_minima(time, data2)
import pdb; pdb.set_trace()
unstable_low2 = r_low[:, minima_low[1]]
c = copy.deepcopy(conn_params)
c.update({'fac_nu_ext_5E': par_transition,
          'fac_nu_ext_6E': 10/3.*par_transition-7/3.,
          'K_stable': 'iteration_1/K_prime.npy'})
network_params = {'connection_params': c,
                  'input_params': input_params}
MAM = MultiAreaModel(network_params, theory=True,
                     theory_spec=theory_params)

S_vector, S, T_vector, T, M = stabilize.S_T(MAM.theory, unstable_low2)
lambda_ev2, u2, v2 = stabilize.eigen_decomp_M(M)

ax.plot(np.real(lambda_ev2), np.imag(lambda_ev2), '.',
        color=myred, markeredgecolor='none', ms=5)


pl.locator_params(axis='y', nbins=3)
pl.locator_params(axis='x', nbins=3)
ax.set_xlim((-18, 5))
ax.set_ylim((-8, 8))
ax.set_xlabel(r'Re$(\lambda)$', labelpad=-0.1)
ax.set_ylabel(r'Im$(\lambda)$', labelpad=5)
ax.vlines(1, -8., 8., linestyles='dashed')


"""
Panel B
"""
data1 = utils.load_iteration(1)
(par_transition, r_low, r_high,
 minima_low, minima_high) = utils.determine_velocity_minima(time, data1)

unstable_low1 = r_low[:, minima_low[1]]
c = copy.deepcopy(conn_params)
c.update({'fac_nu_ext_5E': par_transition,
          'fac_nu_ext_6E': 10/3.*par_transition-7/3.})
network_params = {'connection_params': c,
                  'input_params': input_params}
MAM = MultiAreaModel(network_params, theory=True,
                     theory_spec=theory_params)

# Individual shifts of first iteration
S_vector, S, T_vector, T, M = stabilize.S_T(MAM.theory, unstable_low1)
lambda_ev1, u1, v1 = stabilize.eigen_decomp_M(M)
delta_bar_nu_star = stabilize.fixed_point_shift('fac_nu_ext_5E_6E',
                                                MAM.theory, M_target.theory, unstable_low1)
delta_nu_star = np.dot(np.linalg.inv(np.identity(M.shape[0]) - M), delta_bar_nu_star)

lambda_ev, u, v = stabilize.eigen_decomp_M(M)

a_hat = np.dot(v, delta_bar_nu_star)
v_hat = np.dot(v, unstable_low1)
epsilon = - 1. * a_hat / v_hat

fac = (MAM.theory.NP['tau_syn'] /
       MAM.theory.network.params['neuron_params']['single_neuron_dict']['C_m'])
denom = (S * MAM.theory.network.J_matrix[:, :-1] +
         T * MAM.theory.network.J_matrix[:, :-1]**2) * fac * MAM.theory.NP['tau_m']

individual_deltas1 = []
for ii in range(lambda_ev1.size):
    eigen_proj = np.outer(u[:, ii], v[ii])
    delta_K = epsilon[ii] * eigen_proj / denom
    K_prime = MAM.K_matrix[:, :-1] + np.real(delta_K)
    K_prime[np.where(K_prime < 0.)] = 0.
    individual_deltas1.append(np.sum(np.abs(K_prime - MAM.K_matrix[:, :-1])))
individual_deltas1 = np.array(individual_deltas1)
individual_shifts1 = -1. * a_hat / (1 - lambda_ev1)

indices = np.argsort(np.real(lambda_ev1))[::-1]
deltas = individual_deltas1[indices]
shifts = individual_shifts1[indices]

shift_projections = []
shift_projections_cc = []
self_proj = np.dot(delta_nu_star, delta_nu_star)
for ii in range(254):
    proj = np.dot(u[:, indices][:, ii], delta_nu_star)
    shift_projections.append(shifts[ii] * proj / self_proj)
    shift_projections_cc.append(np.complex.conjugate(
        shifts[ii] * proj / self_proj))


shift_projections = np.array(shift_projections)
shift_projections_cc = np.array(shift_projections_cc)

ax = panel_factory.new_panel(1, 0, r'B', label_position='leftleft')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

eta, cc_indices = np.unique(np.round(0.5 * (shift_projections_cc +
                                            shift_projections), 15),
                            return_index=True)
deltas = deltas[cc_indices]
lambdas = np.real(lambda_ev1[indices][cc_indices])
eta = eta[np.argsort(lambdas)[::-1]]
deltas = deltas[np.argsort(lambdas)[::-1]]

ax.plot(abs(deltas[0]) / np.sum(MAM.K_matrix), abs(eta)
        [0], '.', color=myred, markersize=5)
ax.plot(abs(deltas[1:]) / np.sum(MAM.K_matrix),
        abs(eta)[1:], '.', color='k', markersize=5)
print("Change by critical eigendirection", deltas[0] / np.sum(MAM.K_matrix))
pl.locator_params(axis='y', nbins=3)

ax.set_xlabel(
    r'$ \sum _{l,n}\left| \delta K_{ln} \right|/ \sum_{l,n} K_{ln} $', labelpad=1.)
ax.set_xscale('Log')

ax.set_ylim((-0.1, 1.))
ax.set_xticks(10 ** np.arange(-6, 1., 6))
# ax.xaxis.set_minor_locator(NullLocator())

ax.set_ylabel(r'$| \eta_l |$')
pos = ax.get_position()
inset_position = [pos.x0 - 0.02, pos.y0 + 0.1, 0.2, 0.2]
ax_inset = pl.axes(inset_position, projection='polar')
ax_inset.plot([np.pi / 2., np.pi / 2.], [0., 1.], color='0.5')
for ii in range(shift_projections.size):
    if ii == 0:
        color = myred
        lw = 1
    else:
        color = 'k'
        lw = 0.2

    norm1 = np.linalg.norm(
        individual_shifts1[ii] * u[:, ii])
    norm2 = np.linalg.norm(delta_nu_star)
    angle = np.arccos(np.dot(individual_shifts1[ii] * u[:, ii], delta_nu_star) / (
        norm1 * norm2))
    ax_inset.plot([angle + np.pi / 2., angle + np.pi / 2.],
                  [0., 1.], color=color, linewidth=lw)
    if angle > 2:
        print(angle)
    ax_inset.text(np.pi / 2. - 0.1, 0.4, r'$\delta \nu^{\ast}$')

ax_inset.set_rgrids([1.], visible=False)
ax_inset.set_thetagrids([], visible=False)


"""
Panel C and D
Critical eigenvectors
"""

for i, (lambda_ev, u) in enumerate(zip([lambda_ev1, lambda_ev2], [u1, u2])):
    uc = u[:, np.argmax(np.real(lambda_ev))]
    
    ax = panel_factory.new_panel(i, 1, 'C', label_position=-0.26)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')

    matrix = np.zeros((len(area_list), 8))
    for i, area in enumerate(area_list):
        mask = create_vector_mask(M_base.structure, areas=[area])
        m = np.real(uc[mask])
        if area == 'TH':
            m = np.insert(m, 2, 0.0)
            m = np.insert(m, 2, 0.0)
        matrix[i, :] = m[::-1]
    matrix = np.transpose(matrix)

    im = ax.pcolormesh(matrix, vmin=-1., vmax=1., cmap=cmap2)
    ax.set_xlim((0, 32))
    t = pl.FixedLocator(np.arange(-1., 1.01, 0.5))
    pl.colorbar(im, ticks=t)

    ax.set_xticks([0.5, 3.5, 14.5, 24.5, 28.5])
    ax.xaxis.set_major_locator(pl.FixedLocator([0, 1, 4, 9, 24, 31]))
    ax.xaxis.set_major_formatter(pl.NullFormatter())
    ax.xaxis.set_minor_locator(pl.FixedLocator(
        [0.5, 2.5, 6.5, 16.5, 27.5, 31.5]))
    ax.set_xticklabels([8, 7, 6, 5, 4, 2], minor=True)
    ax.tick_params(axis='x', which='minor', length=0.)
    ax.set_xlabel('Arch. type', labelpad=-0.1)

    y_index = list(range(len(MAM.structure['V1'])))
    y_index = [a + 0.5 for a in y_index]
    
    ax.set_yticks(y_index)
    ax.set_yticklabels(population_labels[::-1])

    if i == 0:
        ax.set_ylabel('Population')
        ax.yaxis.set_label_coords(-0.2, 0.5)

"""
Save figure
"""
pl.savefig('Fig5_eigenspace.eps')
