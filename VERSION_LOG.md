## MAM v1.1.0

### New Features:

* Documentation Enhancements:
  * Streamlined README.md with a Try It On EBRAINS button and step-by-step user instructions.
  * Removed "Testing on EBRAINS" section for clarity.

* Parameter Tuning Improvements:
  * Segregated parameters in down-scale multi-area mode into Parameters to Tune and Default Parameters.
  * Introduced exposure of four user-friendly parameters, while retaining others for internal tuning.

* Visualization Augmentations:
  * Introduced Extract and Visualize Interareal Connectivity to display area-level relative connectivity via heatmaps.
  * Added Simulation Results Visualization section with separate code modules in “./figures/MAM2EBRAINS”.
  * Enriched visualization with three new plots detailing instantaneous firing rate, resting state, and time-averaged population rates.
  * Refined representation of resting state plots inspired from Schmidt M et al. (2018), allowing users flexible area selection, altered synchrony representation, and a focus on binned spike histograms.

### Enhancements:

* Notebook Refinements:
  * Overhauled Jupyter Notebook structure with an accessible table of contents for user navigation.
  * Enhanced parameter descriptions for both exposed and default sets.
  * Incorporated model overview and concise description of the down-scaled multi-area model.
  * Cross-referenced relevant publication figures for user benefit.
  
### Code Optimizations:

* Minimized irrelevant print statements in codebase for clearer user outputs.
* Updated .gitignore to exclude checkpoint files.

### Bug Fixes:

* Resolved file path separator issue in ./multiarea_model/data_multiarea/SLN_logdensities.R.
* Addressed datatype concerns in ./multiarea-model/analysis.py for array initialization.

## MAM v1.0.0

### Bug Fixes:

* Rectified incorrect NEST logo URL in README.md.
