## Figures of Schmidt M, Bakker R, Shen K, Bezgin B, Diesmann M & van Albada SJ (2018) A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque cortex.

[![www.python.org](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org) <a href="http://www.nest-simulator.org"> <img src="https://raw.githubusercontent.com/nest/nest-simulator/master/extras/logos/nest-simulated.png" alt="NEST simulated" width="50"/></a> [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This folder contains the scripts to reproduce all figures of Schmidt M, Bakker R, Shen K, Bezgin B, Diesmann M & van Albada SJ (2018) A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque cortex. (under review)

The figure scripts are named according to their ordering in the paper. To execute them, you can choose to either work with newly simulated data or the original data used in the publication. 

### Original simulation data

You can retrieve the original data from our data repository: [https://web.gin.g-node.org/maximilian.schmidt/multi-area-model-data](https://web.gin.g-node.org/maximilian.schmidt/multi-area-model-data). By default, all plot scripts are configured to use the old data (`LOAD_ORIGINAL_DATA = True` in all scripts). You have to download the data and then define the variable `original_data_path` in `config.py`. 

### Creating new simulation data

To work with newly simulated data, you first have to run the necessary simulations. In `network_simulations.py`, we define the parameter dictionaries for all simulations along with data structures associating simulations with the respective figures. To run the simulations for a specific figure, execute the `run_simulations(figure)` (set `all` for all figures and e.g. `Fig2` for figure 2). Please note that you will need to provide sufficient resources on an HPC cluster.
You then have to set `LOAD_ORIGINAL_DATA = False` in the `Snakefile` and the respective plot scripts to switch to using new data.

### Experimental data

Figure 6 requires the experimental data of Chu CCJ, Chien PF, Hung CP. Multi-electrode recordings of ongoing activity
and responses to parametric stimuli in macaque V1, available in the crcns.org database. Please register on [https://www.crcns.org/](https://www.crcns.org/), download the data from [https://portal.nersc.gov/project/crcns/download/pvc-5](https://portal.nersc.gov/project/crcns/download/pvc-5), extract them into a path and then set this path in `helpers.py`.

Figure 8 requires the experimental fMRI data (described by Babapoor-Farrokhran et al. (2013), see Methods section of Schmidt et al. 2018 for more details) that are contained in this repository in `Fig8_exp_func_conn.csv`.

### Requirements

Reproducing the figures requires some additional Python packages listed in `additional_requirements.txt`. They can be installed using pip by executing:

`pip install -r additional_requirements.txt`

The calculation of BOLD signals from the simulated firing rates for Fig. 8 requires an installation of R and the R library `neuRosim` (<https://cran.r-project.org/web/packages/neuRosim/index.html>).

To produce Figure 8, an installation of the `infomap` package is required. Please follow the instructions on the website http://www.mapequation.org/code.html . After installing `infomap`, please specify the path to the executable in `helpers.py`.

### Snakemake workflow

The entire workflow from raw spike files till the final figures is defined in `Snakefile` and `Snakefile_preprocessing`. If snakemake is installed, the figures can be produced by executing `snakemake`.

<img src="https://raw.githubusercontent.com/nest/nest-simulator/master/extras/logos/nest-simulated.png" alt="NEST simulated" width="200"/>
