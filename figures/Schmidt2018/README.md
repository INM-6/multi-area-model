## Figures of Schmidt M, Bakker R, Hilgetag C-C, Diesmann M and van Albada SJ: "Multi-scale account of the network structure of macaque visual cortex"

This folder contains the scripts to reproduce all figures of Schmidt M, Bakker R, Hilgetag C-C, Diesmann M and van Albada SJ: "Multi-scale account of the network structure of macaque visual cortex", Brain Structure and Function (2018), 223:1409 [https://doi.org/10.1007/s00429-017-1554-4](https://doi.org/10.1007/s00429-017-1554-4)

![Model overview](../../model_construction.png)

Please note that the placement of areas in Figure 7 will deviate from the published figure, because their location depends on the force-directed algorithm implemented in `igraph` and `python-igraph` does not allow manual setting of the random seed for the algorithm. This is a mere visual issue and does not affect the scientific content.

Please note that, since we currently cannot publish the data on Neuronal Densities, Figures 2 and 5 can currently not be produced and executing it throws an error.

Reproducing the figures requires some additional Python packages listed in `additional_requirements.txt`. They can be installed using pip by executing:

`pip install -r additional_requirements.txt`

To produce Figure 7, an installation of the `infomap` package is required. Please follow the instructions on the website http://www.mapequation.org/code.html.

If snakemake is installed, the figures can be produced by executing

`snakemake`

[![www.python.org](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org) <a href="http://www.nest-simulator.org">
