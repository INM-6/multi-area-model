## MAM v1.1.0

This release brings a number of improvements and new features, including the down-scaling multi-area model, which enhance the model's reliability, usability, and overall user experience. It also serves as a base version that specifies the dependencies required for our future releases up to and including MAM v2.0.0.

For detailed changes made in this release since our last release MAM v1.0.0, please check our [version log][https://github.com/INM-6/MAM2EBRAINS/edit/pre-cr-pr/VERSION_LOG.md].

### Enhancements

* The structure of the notebook has been revamped for improved clarity and user navigation.

* Enhanced Parameter Descriptions: Parameters are now accompanied by more detailed descriptions, providing users with a clearer understanding and better control over model configurations.

* Visualization Enhancements: Dive deeper into your simulation results with the addition of new plots that offer richer insights into the model's behavior.

* Try it on EBRAINS: For users looking for a seamless experience, we've integrated a "Try it on EBRAINS" button. Dive right in and experience MAM on the EBRAINS platform!

* Improved Documentation: The README.de file has been updated with comprehensive user instructions to guide both new and returning users through the features and functionalities of MAM.

### Bug fixes

* Corrected the separator from "" to "/" in ./multiarea_model/data_multiarea/SLN_logdensities.R to fix the file path of ./multiarea_model/data_multiarea/bbAlt.R


## MAM v1.0.0

This code implements the spiking network model of macaque visual cortex developed at the Institute of Neuroscience and Medicine (INM-6), Research Center Jülich.

The model has been documented in the following publications:

1. Schmidt M, Bakker R, Hilgetag CC, Diesmann M & van Albada SJ Multi-scale account of the network structure of macaque visual cortex Brain Structure and Function (2018), 223: 1409 https://doi.org/10.1007/s00429-017-1554-4

2. Schuecker J, Schmidt M, van Albada SJ, Diesmann M & Helias M (2017) Fundamental Activity Constraints Lead to Specific Interpretations of the Connectome. PLOS Computational Biology, 13(2): e1005179. https://doi.org/10.1371/journal.pcbi.1005179

3. Schmidt M, Bakker R, Shen K, Bezgin B, Diesmann M & van Albada SJ (2018) A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque cortex. PLOS Computational Biology, 14(9): e1006359. https://doi.org/10.1371/journal.pcbi.1006359

The code in this repository is self-contained and allows one to reproduce the results of all three papers.

The code was improved since the above publications to function with more modern versions of NEST (3+) as well as other minor updates.

### Bug fixes
* corrected the URL of NEST logo in README.md


