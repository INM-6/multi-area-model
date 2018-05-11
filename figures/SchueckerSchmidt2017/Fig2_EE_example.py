import pylab as pl
from plotfuncs import *
import pyx
import os

"""
Panel C/E/F
"""
# set up figure and axis
mrk = 6.
scale = 1.0
width = 0.5 * 4.56  # inches fo JoN single column
n_horz_panels = 1
n_vert_panels = 4
panel_factory = create_fig_JoN(
    1, scale, width, n_horz_panels, n_vert_panels, hoffset=0.2, voffset=+0.28)
axA = panel_factory.new_panel(
    0, 1, '', label_position='leftleft', panel_height_factor=0.95)
pl.locator_params(axis='y', nbins=4)
axC = panel_factory.new_panel(
    0, 2, 'E', label_position='leftleft', voffset=-0.0)
pl.locator_params(axis='y', nbins=4)
axA.spines['right'].set_color('none')
axA.spines['top'].set_color('none')
axA.yaxis.set_ticks_position("left")
axA.xaxis.set_ticks_position("bottom")

axA2 = panel_factory.new_panel(
    0, 0, 'C', label_position='leftleft', voffset=-0.05, panel_height_factor=0.7)
pl.locator_params(axis='y', nbins=4)
axA2.spines['right'].set_color('none')
axA2.spines['top'].set_color('none')
axA2.spines['bottom'].set_color('none')
axA2.yaxis.set_ticks_position("left")
axA2.xaxis.set_ticks_position("none")
axA2.set_xticks([])

axC.spines['right'].set_color('none')
axC.spines['top'].set_color('none')
axC.yaxis.set_ticks_position("left")
axC.xaxis.set_ticks_position("bottom")
axB = panel_factory.new_panel(
    0, 3, 'F', label_position='leftleft', voffset=-0.0)
pl.locator_params(axis='y', nbins=4)
axB.spines['right'].set_color('none')
axB.spines['top'].set_color('none')
axB.yaxis.set_ticks_position("left")
axB.xaxis.set_ticks_position("bottom")

# execute plot script
exec(compile(open('EE_example_CEF.py').read(), 'EE_example_CEF.py', 'exec'))

# save figure
pl.savefig('EE_example_CEF.eps')
pl.clf()
pl.close()

"""
Panel D/G
"""
# set up figure and axis
mrk = 6.
scale = 1.0
width = 0.5 * 4.56
n_horz_panels = 1
n_vert_panels = 2
panel_factory = create_fig_JoN(
    1, scale, width, n_horz_panels, n_vert_panels, aspect_ratio_1=True, hoffset=0.2)
axD = panel_factory.new_panel(0, 0, 'D', label_position='leftleft')
axD.spines['right'].set_color('none')
axD.spines['top'].set_color('none')
axD.yaxis.set_ticks_position("left")
axD.xaxis.set_ticks_position("bottom")
axG = panel_factory.new_panel(0, 1, 'G', label_position='leftleft')
axG.spines['right'].set_color('none')
axG.spines['top'].set_color('none')
axG.yaxis.set_ticks_position("left")
axG.xaxis.set_ticks_position("bottom")
pl.locator_params(axis='y', nbins=4)
pl.locator_params(axis='x', nbins=4)

# execute plot script
exec(compile(open('EE_example_DG.py').read(), 'EE_example_DG.py', 'exec'))

# save figure
pl.savefig('EE_example_DG.eps')


"""
merge panels into one figure using pyx
"""

# pyx.text.set(mode='latex')
# pyx.text.preamble(r"\usepackage{helvet}")
c = pyx.canvas.canvas()
#c.text(6.1, 13.7, r'\textbf{\textsf{B}}')
c.insert(pyx.epsfile.epsfile(0, 0., "EE_example_CEF.eps"))
c.insert(pyx.epsfile.epsfile(5.8, 0.05, "EE_example_DG.eps"))
c.insert(pyx.epsfile.epsfile(0.3, 12.3, "EE_example_A.eps", width=4.))
c.insert(pyx.epsfile.epsfile(6.1, 11.0, "EE_example_B.eps", width=5.))
#c.text(0.3, 13.7, r'\textbf{\textsf{A}}')
c.writeEPSfile("fig2.eps")
