## MAM v1.1.0

### New features

* Improved documentation: added in the README.md file the Try It On EBRAINS button and clear and detailed User instruction for users to be able to follow step-by-step instructions without much background knowledge or experience, delete section Testing on EBRAINS
  
* In down-scale multi-area mode, separated all external parameters to Parameters to tun and Default parameters. Parameters to tune consist of 4 parameters we decided to expose to users initially, and default parameters will be tuned by us and are not recommended for users to change

* Added section Extract and visualize interareal connectivity which plots the area-level relative connectivity as heatmaps. Two heatmaps represent the interareal connectivity of full-scale multi-area model (left) and down-scale multi-area model (right). There are small differences between them although we’re calculating relative connectivity as there’s randomness

* Added section Simulation Results Visualization. The code is written in separate modules saved as .py files in “./figures/MAM2EBRAINS” to avoid displaying contents that are not relevant to users

* Added 3 plots in the section Simulation Results Visualization
  * 3.1. Instantaneous and mean firing rate across all populations (existed in MAM v1.0.0, refined in MAM v1.1.0)
  * 3.2 Resting state plots
  * 3.3 Time-averaged population rates

* The 3.2 Resting state plots figure is plotted based on Fig 5. of the paper Schmidt M, Bakker R, Shen K, Bezgin B, Diesmann M & van Albada SJ (2018) A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque cortex. PLOS Computational Biology, 14(9): e1006359. https://doi.org/10.1371/journal.pcbi.1006359, yet there are a few differences:
  * This plot provides the option for users to choose 3 areas to plot the raster plots instead of fixing  V1, V2, and FEF to plot
  * The subplot E Correlation coefficient is replaced as Synchrony
  * The subplot G only plots the binned spike histograms (gray), not the convolved histograms (black)

### Enhancements

* Reconstructed the Jupyter Notebook and added Notebook structure as table of contents that enables users to navigate quickly and easily between different sections. (see the notebook structure for details)
  
* Added detailed and easy-to-understand descriptions to the 4 exposed parameters and also brief comments for the default parameters

* Added the model overview diagram and a short description of the down-scaled multi-area model at the beginning of the jupyter notebook 

* Added descriptions of comparable figures in our publications whenever available so that users can compare the down-scaled model with their costumed parameters and the full-scaled model presented in the paper

* Removed unnecessary print statements in ./multiarea_model/analysis.py and ./multiarea_model/analysis_helpers.py to avoid multiple print that are not relevant to users

* Updated ./.gitignore file to ignore checkpoint files

### Bug fixes

* Corrected the separator from "" to "/" in ./multiarea_model/data_multiarea/SLN_logdensities.R to fix the file path of ./multiarea_model/data_multiarea/bbAlt.R
  
* Fixed bugs in ./multiarea-model/analysis.py: change np.nan*np.ones(params['t_max'] - params['t_min']) to np.nan*np.ones(int(params['t_max'] - params['t_min']))


## MAM v1.0.0

### Bug fixes
* Corrected the URL of NEST logo in README.md


