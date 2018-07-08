import copy
import numpy as np
import pylab as pl
import pyx

from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from plotcolors import myred, myblue
from Fig2_EE_network import network1D, network2D

"""
Figure layout
"""
scale = 1.

# resolution of figures in dpi
# does not influence eps output
pl.rcParams['figure.dpi'] = 300

# font
pl.rcParams['font.size'] = scale * 8
pl.rcParams['legend.fontsize'] = scale * 8
pl.rcParams['font.family'] = "sans-serif"

pl.rcParams['lines.linewidth'] = scale * 1.0

# size of markers (points in point plots)
pl.rcParams['lines.markersize'] = scale * 3.0
pl.rcParams['patch.linewidth'] = scale * 1.0
pl.rcParams['axes.linewidth'] = scale * 1.0     # edge linewidth

# ticks distances
pl.rcParams['xtick.major.size'] = scale * 4      # major tick size in points
pl.rcParams['xtick.minor.size'] = scale * 2      # minor tick size in points
pl.rcParams['lines.markeredgewidth'] = scale * 0.5  # line width of ticks
pl.rcParams['grid.linewidth'] = scale * 0.5
# distance to major tick label in points
pl.rcParams['xtick.major.pad'] = scale * 4
# distance to the minor tick label in points
pl.rcParams['xtick.minor.pad'] = scale * 4
pl.rcParams['ytick.major.size'] = scale * 4      # major tick size in points
pl.rcParams['ytick.minor.size'] = scale * 2      # minor tick size in points
# distance to major tick label in points
pl.rcParams['ytick.major.pad'] = scale * 4
# distance to the minor tick label in points
pl.rcParams['ytick.minor.pad'] = scale * 4

# ticks textsize
pl.rcParams['ytick.labelsize'] = scale * 8
pl.rcParams['xtick.labelsize'] = scale * 8

# use latex to generate the labels in plots
pl.rcParams['text.usetex'] = True
pl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

pl.rcParams['ps.useafm'] = False   # use of afm fonts, results in small files
# Output Type 3 (Type3) or Type 42 (TrueType)
pl.rcParams['ps.fonttype'] = 3


nrows = 2.
ncols = 2.
width = 4.61  # inches for 1.5 JoN columns
height = 5.67
pl.rcParams['figure.figsize'] = (width, height)

fig = pl.figure()

axes = {}

gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.1, right=0.5, top=0.95, bottom=0.9, hspace=0.6, wspace=0.4)
axes['A'] = pl.subplot(gs1[0, :])

gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.1, right=0.5, top=0.87, bottom=0.75, hspace=0.6, wspace=0.4)
axes['C'] = pl.subplot(gs1[0, :])


gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.1, right=0.5, top=0.7, bottom=0.53, hspace=0.6, wspace=0.4)
axes['C2'] = pl.subplot(gs1[0, :])

gs1 = gridspec.GridSpec(2, 1)
gs1.update(left=0.1, right=0.5, top=0.45, bottom=0.075, hspace=0.4, wspace=0.4)
axes['E'] = pl.subplot(gs1[0, :])
axes['F'] = pl.subplot(gs1[1, :])


gs1 = gridspec.GridSpec(1, 1)
gs1.update(left=0.65, right=0.95, top=0.95, bottom=0.8, hspace=0.4, wspace=0.4)
axes['B'] = pl.subplot(gs1[0, :])

gs1 = gridspec.GridSpec(2, 1)
gs1.update(left=0.65, right=0.95, top=0.7,
           bottom=0.075, hspace=0.4, wspace=0.4)
axes['D'] = pl.subplot(gs1[0, :])
axes['G'] = pl.subplot(gs1[1, :])

for ax in axes.values():
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

rate_exts_array = np.arange(150., 170.1, 1.)

network_params = {'K': 420.,
                  'W': 10.}

for label in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    pl.text(-0.17, 1.05, r'\bfseries{}' + label,
            fontdict={'fontsize': 10.,
                      'weight': 'bold',
                      'horizontalalignment': 'left',
                      'verticalalignment': 'bottom'},
            transform=axes[label].transAxes)

"""
1D network
"""


ax = axes['A']
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])

ax_inset = pl.axes([0.39, .76, 0.1, .05])
ax_inset.yaxis.set_ticks_position('none')
ax_inset.xaxis.set_ticks_position('none')
ax_inset.set_xticks([0, 10000])
ax_inset.set_xticklabels([0, r'$10^5$'])
ax_inset.set_xlim([0., 10000])
ax_inset.set_yticks([0, 500])
ax_inset.set_ylim([0, 500])
ax_inset.tick_params(axis='x', labelsize=4, pad=1)
ax_inset.tick_params(axis='y', labelsize=4, pad=1)


x = np.arange(0, 150., 1.)


"""
Panel C (top): Transfer function
for three cases:
- normal network
- noiseless neurons
- neurons without refractory period
"""
ax = axes['C']
ax.xaxis.set_ticks_position('none')
ax.spines['bottom'].set_color('none')
ax.set_yticks([0., 50.])

# Normal network with rate_ext = 160.
input_params = {'rate_ext': 160.}
network_params.update({'input_params': input_params})
net = network1D(network_params)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
ax.plot(x, y, '0.3')

x_long = np.arange(0, 100000., 500.)
y_long = np.fromiter([net.Phi(x_long[j])[0]
                      for j in range(len(x_long))], dtype=np.float)
ax_inset.plot(x_long, y_long, '0.3')

# Normal network with rate_ext = 160. without refractory period
input_params = {'rate_ext': 160.}
network_params2 = copy.deepcopy(network_params)
network_params2.update(
    {'neuron_params': {'single_neuron_dict': {'t_ref': 0.}}})
net = network1D(network_params2)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
ax.plot(x, y, color=myred)

# Noisefree case
input_params = {'rate_ext': 160.}
network_params.update({'input_params': input_params})
net = network1D(network_params)
NP = net.params['neuron_params']['single_neuron_dict']
y = np.fromiter([net.Phi_noisefree(x[j])
                 for j in range(len(x))], dtype=np.float)

ax.plot(x, y, color=myblue)

ax.plot(x, x, '--', color='k')

ax.set_xticks([])
ax.set_xlim([-3, 70])
ax.set_ylim([-3, 70])


"""
Panel C (bottom): Transfer function
for three levels of external stimulation.
"""
ax = axes['C2']
colors = ['k', '0.3', '0.7']
markers = ['d', '+', '.']

for i, rate_ext in enumerate([150., 160., 170.]):
    input_params = {'rate_ext': rate_ext}
    network_params.update({'input_params': input_params})
    net2 = network1D(network_params)
    y = np.fromiter([net2.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
    ax.plot(x, y, colors[i])
    # Plot fixed points
    ind = np.where(np.abs(y - x) < 0.2)
    for j in ind[0]:
        if x[j] < 3.:
            ax.plot(x[j], y[j], 'd', ms=3, color='k')
        elif x[j] > 3. and x[j] < 28.8:
            ax.plot(x[j], y[j], '.', ms=7, color='k')
        else:
            ax.plot(x[j], y[j], '+', ms=5, color='k')

ax.plot(x, x, '--', color='k')
ax.set_xlabel(r'Rate $\nu\quad(1/\mathrm{s})$')
ax.set_ylabel(r'Rate $\Phi(\nu)\quad(1/\mathrm{s})$')
ax.set_xlim([-3, 70])
ax.set_ylim([-3, 70])
ax.set_yticks([0., 50.])
acb = pl.axes([-0.1, .59, 0.55, 0.07])
acb.axis('off')
cmap = pl.get_cmap('Greys_r')
line_colors = cmap(np.linspace(0, 1, 4)[:3])
cmap2 = ListedColormap(line_colors)
sm = pl.cm.ScalarMappable(
    cmap=cmap2, norm=pl.Normalize(vmin=150., vmax=170.))
sm.set_array([])
cbticks = [170., 150.]
cbar = pl.colorbar(sm, ax=acb, ticks=cbticks, shrink=0.5, aspect=10, pad=-0.15,
                   anchor=(1., 0.5), orientation='horizontal')
cbar.set_label(r'$\nu_{\mathrm{ext}}$', labelpad=-2)
cbar.ax.invert_xaxis()
cbar.solids.set_edgecolor('none')

"""
Panel E: Fixed points of 1D system
"""
rate_init = np.arange(0., 100., 10.)
ax = axes['E']

for rate_ext in np.arange(150., 170., 1.):
    fp_list = []
    input_params = {'rate_ext': rate_ext}
    network_params.update({'input_params': input_params})
    net = network1D(network_params)
    for init in rate_init:
        res = net.fsolve([init])
        if res['eps'] == 'The solution converged.':
            fp_list.append(res['rates'][0][0])
    for fp in fp_list:
        if fp < 3.:
            ax.plot(rate_ext, fp, 'D', color='k', markersize=2)
        elif fp > 3. and fp < 28.8:
            ax.plot(rate_ext, fp, '.', color='k', markersize=3)
        elif fp > 28.8:
            ax.plot(rate_ext, fp, '+', color='k', markersize=3)
ax.set_xlim((150., 170.))
ax.set_yticks([0., 50.])
ax.set_xlabel(r'$\nu_{\mathrm{ext}}\quad(1/\mathrm{s})$')
ax.set_ylabel(r'Rate $\nu\quad(1/\mathrm{s})$')

"""
Panel F: Flux in the bistable case
"""
ax = axes['F']
a = pl.axes([0.27, .1, 0.12, .06])
a.set_xticks([0, 0.03])
a.set_xlim([0., 0.03])
a.set_yticks([-0.02, 0.02])
a.set_ylim([-0.02, 0.02])
a.tick_params(axis='x', labelsize=4, pad=2)
a.tick_params(axis='y', labelsize=4, pad=1)


network_params_base = {'K': 420.,
                       'W': 10.,
                       'input_params': {'rate_ext': 160.}}
network_params_inc = {'K': 420.,
                      'W': 10.,
                      'input_params': {'rate_ext': 161.}}
network_params_stab = {'K': 420.,
                       'W': 10.,
                       'input_params': {'rate_ext': 161.}}

# Normal network with rate_ext = 160.
net = network1D(network_params_base)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
ax.plot(x, y - x, color='k')
a.plot(x, y - x, color='k')
ax.hlines(0., 0., 70., linestyles='dashed')
a.hlines(0., 0., 70., linestyles='dashed')
fp_base = net.fsolve(([18.]))['rates'][0][0]

# Normal network with rate_ext = 161.
net = network1D(network_params_inc)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
ax.plot(x, y - x, color=myblue)
a.plot(x, y - x, color=myblue, lw=4.)
fp_inc = net.fsolve(([18.]))['rates'][0][0]

# Stabilized network with rate_ext = 161.
deltaK = -1. * network_params['K'] * (161. - 160.) / fp_base
print(network_params['K'] + deltaK)
network_params_stab.update({'K_stable': network_params['K'] + deltaK})
print(network_params_stab)
net = network1D(network_params_stab)
y = np.fromiter([net.Phi(x[j])[0] for j in range(len(x))], dtype=np.float)
ax.plot(x, y - x, color=myred)
a.plot(x, y - x, color=myred)
fp_s = net.fsolve(([18.]))['rates'][0][0]

ax.hlines(0., 0., 70., linestyles='dashed')


ax.set_xlabel(r'Rate $\nu\quad(1/\mathrm{s})$')
ax.set_ylabel(r'Flux $\dot\nu\quad(1/\mathrm{s})$', labelpad=0)
ax.set_xlim([0, 50])
ax.set_ylim([-10, 7])

ylim = 7.
y0 = ylim
x0 = 0.
height = 0.5
rect_base = pl.Rectangle((x0, y0 - 1.1 * height), width=fp_base,
                         height=height, fill=True, color='black')
rect_2 = pl.Rectangle((x0, y0 - 2.7 * height),
                      width=fp_inc, height=height, fill=True, color=myblue)
rect_s = pl.Rectangle((x0, y0 - 4.3 * height),
                      width=fp_s, height=height, fill=True, color=myred)
ax.add_patch(rect_base)
ax.add_patch(rect_2)
ax.add_patch(rect_s)


"""
2D network
"""
ax = axes['B']
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])

"""
Panel D and G: Phase space with flux and nullclines
"""
ax = axes['D']
axG = axes['G']
axG.set_ylabel(r'Excitatory rate 1 $(10^{-2}/\mathrm{s})$')
axG.set_xlabel(r'Excitatory rate 2 $(10^{-2}/\mathrm{s})$')

rates_init = np.arange(0., 200., 10.)

network_params_base = {'K': 420.,
                       'W': 10.,
                       'input_params': {'rate_ext': 160.}}
network_params_inc = {'K': 420.,
                      'W': 10.,
                      'input_params': {'rate_ext': 161.}}
network_params_stab = {'K': 420.,
                       'W': 10.,
                       'input_params': {'rate_ext': 161.}}

colors = ['k', myblue, myred]
for i, netp in enumerate([network_params_base,
                          network_params_inc,
                          network_params_stab]):
    net = network2D(netp)

    # For the stabilized network, compute fixed point of base network
    # and adapt indegree to stabilize the network
    if i == 0:
        fp = net.fsolve(([18., 18.]))['rates'][0][0]
        deltaK = -1. * network_params['K'] * (161. - 160.) / fp
        network_params_stab.update({'K_stable': network_params['K'] + deltaK})

    fp_list = []
    for init in rate_init:
        res = net.fsolve([init, init])
        if res['eps'] == 'The solution converged.':
            fp_list.append(res['rates'][0])
    print(fp_list)
    for fp in fp_list:
        if fp[0] < 3.:
            if i == 1:
                ax.plot(fp[0], fp[1], 's', color=colors[i], markersize=2 + 3)
                axG.plot(fp[0] * 1e2, fp[1] * 1e2, 's',
                         color=colors[i], markersize=2 + 3)
            else:
                ax.plot(fp[0], fp[1], 'D', color=colors[i], markersize=2)
                axG.plot(fp[0] * 1e2, fp[1] * 1e2, 'D',
                         color=colors[i], markersize=2)
        elif fp[0] > 3. and fp[0] < 28.8:
            ax.plot(fp[0], fp[1], '.', color=colors[i], markersize=5)
            axG.plot(fp[0] * 1e2, fp[1] * 1e2, '.',
                     color=colors[i], markersize=5)
            separatrix = ([0, 2 * fp[0]], [2 * fp[1], 0])
            if i == 0:
                ax.plot(separatrix[0], separatrix[1], lw=5, color=colors[i])
            else:
                ax.plot(separatrix[0], separatrix[1], color=colors[i])

        elif fp[0] > 28.8:
            if i == 0:
                ax.plot(fp[0], fp[1], '+', color=colors[i], markersize=3)
                axG.plot(fp[0] * 1e2, fp[1] * 1e2, '+',
                         color=colors[i], markersize=3)


ax.set_ylabel(r'Excitatory rate 1 $(1/\mathrm{s})$')
ax.set_xlabel(r'Excitatory rate 2 $(1/\mathrm{s})$')
y0 = 0.
x1 = 0.015
y1 = 0.015

# vector fields
netp = network_params_base
net = network2D(netp)
range_vector = np.arange(0., 51., 10.)
x, y, vx, vy = net.vector_field(range_vector, range_vector)
ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=8)

range_vector = np.linspace(0.001, 0.015, 4)
x, y, vx, vy = net.vector_field(range_vector, range_vector)
axG.quiver(x * 1e2, y * 1e2, vx * 1e2, vy * 1e2,
           angles='xy', scale_units='xy', scale=10)

netp = network_params_stab
net = network2D(netp)
range_vector = np.linspace(0.001, 0.015, 4)
x, y, vx, vy = net.vector_field(range_vector, range_vector)
axG.quiver(x * 1e2, y * 1e2, vx * 1e2, vy * 1e2,
           angles='xy', scale_units='xy', scale=10, color=myred)

# nullclines
netp = network_params_base
net = network2D(netp)
x0_vec = np.arange(0, 50, 0.1)
nullcline_x0 = net.nullclines_x0(x0_vec)
ax.plot(nullcline_x0, x0_vec, '--', color='black', label='x0')
ax.plot(x0_vec, nullcline_x0, '--', color='black', label='x0')

# set plot limits
axG.set_xlim((x0 * 1e2, x1 * 1e2))
axG.set_ylim((y0 * 1e2, y1 * 1e2))
ax.set_xlim([-5, 50])
ax.set_ylim([-5, 50])


"""
Save and merge figure
"""
pl.savefig('Fig2_EE_example_mpl.eps')

c = pyx.canvas.canvas()

c.insert(pyx.epsfile.epsfile(0, 0., "Fig2_EE_example_mpl.eps"))
c.insert(pyx.epsfile.epsfile(2., 12.8, "Fig2_EE_example_A.eps", width=2.))
c.insert(pyx.epsfile.epsfile(7.6, 11.0, "Fig2_EE_example_B.eps", width=3.))

c.writeEPSfile("Fig2_EE_example.eps")
