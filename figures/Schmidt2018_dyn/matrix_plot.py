import matplotlib.pyplot as plt
import numpy as np
from plotcolors import myblue, myblue2, myred, myred2
from matplotlib.colors import LogNorm
from helpers import population_labels
from matplotlib import rc_file
rc_file('plotstyle.rc')


def matrix_plot(fig, ax, matrix, position='right'):
    ''' Create a matrix plot of pop. rates for the stabilization manuscript
    '''
    ax.yaxis.set_ticks_position('none')
    cm = plt.get_cmap('rainbow')
    cm = cm.from_list('mycmap', [myblue, myblue2,
                                 'white', myred2,  myred], N=256)
    masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    cm.set_under('0.3')
    cm.set_bad('k')
    im = ax.pcolormesh(masked_matrix, norm=LogNorm(
        vmin=0.001, vmax=500.), cmap=cm)
    ax.set_xlim(0, 32)

    ax.set_xticks([0.5, 3.5, 14.5, 24.5, 28.5])
    ax.xaxis.set_major_locator(plt.FixedLocator([0, 1, 4, 9, 24, 31]))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.FixedLocator([0.5, 2.5, 6.5, 16.5,
                                                 27.5, 31.5]))
    ax.set_xticklabels([8, 7, 6, 5, 4, 2], minor=True, size=8)
    ax.tick_params(axis='x', which='minor', length=0.)

    ax.set_xlabel('Architectural type', labelpad=-0.2)

    y_index = list(range(8))
    y_index = [a + 0.5 for a in y_index]
    t = plt.FixedLocator([0.01, 0.1, 1., 10., 100., 500.])
    cb = plt.colorbar(im, ticks=t, ax=ax)

    if position == 'left':
        ax.set_yticklabels(population_labels, size=8)
        ax.set_yticks(y_index[::-1])
        ax.set_ylabel('Population', labelpad=-0.1)
        cb.remove()
    elif position == 'center':
        ax.set_yticks([])
        cb.remove()
    elif position == 'right':
        ax.set_yticks([])
        ax.text(45., 6, r'$\nu (\mathrm{spikes/s})$', rotation=90)
    if position == 'single':
        ax.set_yticklabels(population_labels)
        ax.set_yticks(y_index[::-1])
        ax.set_ylabel('Population', labelpad=-0.1)
        ax.text(45., 6, r'$\nu (\mathrm{spikes/s})$', rotation=90)


def rate_histogram_plot(fig, ax_pos, matrix, position):
    # set up lower axis

    ax2_pos = (ax_pos[0],
               ax_pos[1] + 0.04,
               1 / 5. * ax_pos[2],
               2 / 6. * ax_pos[3])

    ax1_pos = (ax_pos[0] + 1.2 / 5. * ax_pos[2],
               ax_pos[1] + 0.04,
               2.8 / 5. * ax_pos[2],
               2 / 6. * ax_pos[3])

    ax = plt.axes(ax1_pos)
    plt.locator_params(axis='y', nbins=3)
    ax_2 = plt.axes(ax2_pos)
    plt.locator_params(axis='y', nbins=3)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='minor', length=0.)
    ax.tick_params(axis='y', length=0.)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.set_xscale('Log')
    ax.set_xlabel(r'$\nu (\mathrm{spikes/s})$')
    ax.xaxis.set_label_coords(0.5, -0.8)
    ax.set_xticks(10.**(np.array([-3, -1, 1])))
    ax.set_xticklabels([r'$10^{-3}$', r'$10^{-1}$',
                        r'$10^{1}$'], rotation=20, size=8)
    ax.set_yticks([])
    ax.set_xlim(10**-4, 10**3)
    d = 0.02
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, d), (-5 * d, 5 * d), **kwargs)
    ax.plot((-d - 0.05, d - 0.05), (-5 * d, 5 * d), **kwargs)

    ax_2.yaxis.set_ticks_position('left')
    ax_2.xaxis.set_ticks_position('bottom')
    ax_2.tick_params(axis='x', which='minor', length=0.)
    ax_2.spines['right'].set_color('none')
    ax_2.spines['top'].set_color('none')
    ax_2.set_xscale('Log')
    ax_2.set_xticks(10.**(np.array([-5])))
    ax_2.set_xticklabels([r'$0.0$'], rotation=20, size=8)
    ax_2.tick_params(axis='x', which='major', pad=4)
    ax_2.set_xlim(10**-6, 10**-4)
    ax_2.set_yticks([])
    if position == 'left':
        ax_2.set_ylabel('Count')
        ax_2.yaxis.set_label_coords(-.1, 1.4)

    # set up upper axis

    ax_upper_pos = (ax_pos[0] + 1.2 / 5. * ax_pos[2],
                    ax_pos[1] + 3 / 5. * ax_pos[3] + 0.01,
                    2.8 / 5. * ax_pos[2] + 0.02,
                    2 / 6. * ax_pos[3])

    ax_upper_pos_2 = (ax_pos[0],
                      ax_pos[1] + 3 / 5. * ax_pos[3] + 0.01,
                      1. / 5. * ax_pos[2],
                      2 / 6. * ax_pos[3])

    ax_upper = plt.axes(ax_upper_pos)
    ax_upper_2 = plt.axes(ax_upper_pos_2)
    plt.locator_params(axis='y', nbins=3)

    ax_upper.yaxis.set_ticks_position('left')
    ax_upper.xaxis.set_ticks_position('bottom')
    ax_upper.tick_params(axis='x', which='minor', length=0.)
    ax_upper.tick_params(axis='y', length=0.)
    ax_upper.spines['right'].set_color('none')
    ax_upper.spines['top'].set_color('none')
    ax_upper.spines['left'].set_color('none')
    ax_upper.tick_params(axis='y', length=0.)
    ax_upper.set_xscale('Log')
    ax_upper.set_xticks(10.**(np.array([-3, -1, 1])))
    ax_upper.set_xticklabels([])
    ax_upper.set_yticks([])
    ax_upper.set_xlim(10**-4, 10**3)

    d = 0.02
    kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
    ax_upper.plot((-d, d), (-5 * d, 5 * d), **kwargs)
    ax_upper.plot((-d - 0.05, d - 0.05), (-5 * d, 5 * d), **kwargs)

    ax_upper_2.yaxis.set_ticks_position('left')
    ax_upper_2.xaxis.set_ticks_position('bottom')
    ax_upper_2.tick_params(axis='x', which='minor', length=0.)
    ax_upper_2.spines['right'].set_color('none')
    ax_upper_2.spines['top'].set_color('none')
    ax_upper_2.set_xscale('Log')
    ax_upper_2.set_xticks(10.**(np.array([-5])))
    ax_upper_2.set_xticklabels([])
    ax_upper_2.set_yticks([])
    ax_upper_2.set_xlim(10**-6, 10**-4)

    # Plot of rates

    E_rates = matrix[1::2]
    bins = 10**np.arange(-6., 3., 0.15)
    E_vals, E_bins = np.histogram(
        E_rates, bins=bins, range=(bins.min(), bins.max()))

    I_rates = matrix[::2]
    bins = 10**np.arange(-6., 3., 0.15)
    I_vals, I_bins = np.histogram(
        I_rates, bins=bins, range=(bins.min(), bins.max()))

    ax.bar(I_bins[:-1], I_vals, width=np.diff(I_bins),
           color=myred, linewidth=0.4, edgecolor='none')
    ax_2.bar(I_bins[:-1], I_vals, width=np.diff(I_bins),
             color=myred, linewidth=0.4, edgecolor='none')
    ax_upper.bar(E_bins[:-1], E_vals, width=np.diff(E_bins),
                 color=myblue, linewidth=0.4, edgecolor='none')
    ax_upper_2.bar(E_bins[:-1], E_vals, width=np.diff(E_bins),
                   color=myblue, linewidth=0.4, edgecolor='none')
