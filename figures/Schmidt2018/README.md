## Figures of Schmidt M, Bakker R, Hilgetag C-C, Diesmann M and van Albada SJ: "Multi-scale account of the network structure of macaque visual cortex"

This folder contains the scripts to reproduce all figures of Schmidt M, Bakker R, Hilgetag C-C, Diesmann M and van Albada SJ: "Multi-scale account of the network structure of macaque visual cortex", Brain Structure and Function (2018), 223:1409 [https://doi.org/10.1007/s00429-017-1554-4](https://doi.org/10.1007/s00429-017-1554-4)

Please note: Figures 2, 5, and 8 show slight deviations from the published figures in the paper. Published Figures 2 and 5 miss a few data points. This error slipped in during the review process. Importantly, the presented fits are identical in the (correct) figures in this repository and in the manuscript. These deviations thus do not affect the scientific conclusions.

Please note that the placement of areas in Figure 7 will deviate from the published figure, because their location depends on the force-directed algorithm implemented in `igraph` and `python-igraph` does not allow manual setting of the random seed for the algorithm. This is a mere visual issue and does not affect the scientific content.

Please note that, since we currently cannot publish the data on Neuronal Densities, Figure 2 can currently not be produced and executing it throws an error.

If snakemake is installed, the figures can be produced by executing

`snakemake`
