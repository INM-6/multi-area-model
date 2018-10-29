import numpy as np
import pylab as pl


class panel_factory():
    """Class that generates a subpanel in the PloS figure"""

    def __init__(self, scale, figure, n_pan_x, n_pan_y, hoffset,
                 voffset, hspace, vspace, panel_width, panel_height):
        self.scale = scale
        self.figure = figure
        self.n_pan_x = n_pan_x
        self.n_pan_y = n_pan_y
        self.voffset = voffset
        self.hoffset = hoffset
        self.voffset = voffset
        self.hspace = hspace
        self.vspace = vspace
        self.panel_width = panel_width
        self.panel_height = panel_height

    def new_panel(self, nx, ny, label, label_position='center', voffset=0., panel_height_factor=1.):
        """Create new panel with an axes object at position nx, ny"""

        assert(nx >= 0 and nx < self.n_pan_x)
        assert(ny >= 0 and ny < self.n_pan_y)

        pos = [self.hoffset + nx * (self.hspace + self.panel_width),
               voffset + self.voffset +
               (self.n_pan_y - ny - 1) * (self.vspace + self.panel_height),
               self.panel_width, self.panel_height * panel_height_factor]

        # generate axes object
        ax = pl.axes(pos)

        # panel labels
        if label != '':
            # workaround to adjust label both for vertically and horizontally aligned panels
            if self.panel_width > self.panel_height:
                y = 1.03
            else:
                y = 1.01
            if label_position == 'center':
                # position of panel label (relative to each subpanel)
                label_pos = [0.33, y]
            elif label_position == 'left':
                # position of panel label (relative to each subpanel)
                label_pos = [0.0, y]
            elif label_position == 'leftleft':
                # position of panel label (relative to each subpanel)
                label_pos = [-0.2, y]
            elif type(label_position) == float:
                # position of panel label (relative to each subpanel)
                label_pos = [label_position, y]

            label_fs = self.scale * 10           # fontsize of panel label

            pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
                    fontdict={'fontsize': label_fs,
                              'weight': 'bold',
                              'horizontalalignment': 'left',
                              'verticalalignment': 'bottom'},
                    transform=ax.transAxes)

        return ax

    def new_empty_panel(self, nx, ny, label, label_position='left'):
        """Create new panel at position nx, ny"""

        assert(nx >= 0 and nx < self.n_pan_x)
        assert(ny >= 0 and ny < self.n_pan_y)

        pos = [self.hoffset + nx * (self.hspace + self.panel_width),
               self.voffset + (self.n_pan_y - ny - 1) *
               (self.vspace + self.panel_height),
               self.panel_width, self.panel_height]

        ax = pl.axes(pos, frameon=False)
        pl.xticks([])
        pl.yticks([])

        # panel label
        if label != '':
            # workaround to adjust label both for vertically and horizontally aligned panels
            if self.panel_width > self.panel_height:
                y = 1.03
            else:
                y = 1.01
            if label_position == 'center':
                # position of panel label (relative to each subpanel)
                label_pos = [0.33, y]
            elif label_position == 'left':
                # position of panel label (relative to each subpanel)
                label_pos = [0.0, y]
            elif label_position == 'leftleft':
                # position of panel label (relative to each subpanel)
                label_pos = [-0.2, y]
            elif type(label_position) == float:
                # position of panel label (relative to each subpanel)
                label_pos = [label_position, y]
            label_fs = self.scale * 10           # fontsize of panel label

            pl.text(label_pos[0], label_pos[1], r'\bfseries{}' + label,
                    fontdict={'fontsize': label_fs,
                              'weight': 'bold',
                              'horizontalalignment': 'left',
                              'verticalalignment': 'bottom'},
                    transform=ax.transAxes)

        return pos


def create_fig(fig, scale, width, n_horz_panels, n_vert_panels,
               hoffset=0.1, voffset=0.18, squeeze=0.25, aspect_ratio_1=False):
    """Create JoN figure"""

    panel_wh_ratio = (1. + np.sqrt(5)) / 2.  # golden ratio
    if aspect_ratio_1:
        panel_wh_ratio = 1

    height = width / panel_wh_ratio * n_vert_panels / n_horz_panels

    pl.rcParams['figure.figsize'] = (width, height)

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
    # not needed anymore in newer versions
    # using this, font detection fails on adobe illustrator 2010-07-20
    pl.rcParams['text.usetex'] = True
    pl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    pl.rcParams['ps.useafm'] = False   # use of afm fonts, results in small files
    # Output Type 3 (Type3) or Type 42 (TrueType)
    pl.rcParams['ps.fonttype'] = 3

    pl.rcParams['svg.fonttype'] = 'path'
    fig = pl.figure(fig)
    pl.clf()

    panel_width = 1. / n_horz_panels * 0.75   # relative width of each subpanel
    # horizontal space between subpanels
    hspace = 1. / n_horz_panels * squeeze

    panel_height = 1. / n_vert_panels * 0.70   # relative height of each subpanel
    vspace = 1. / n_vert_panels * 0.25         # vertical space between subpanels

    # left margin (relative coordinates)
    hoffset = hoffset
    # bottom margin (relative coordinates)
    voffset = voffset / n_vert_panels

    return panel_factory(scale, fig, n_horz_panels, n_vert_panels,
                         hoffset, voffset, hspace, vspace, panel_width, panel_height)
