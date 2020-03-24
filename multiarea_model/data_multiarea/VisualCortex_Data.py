"""
VisualCortexData
================

This script provides the function process_raw_data which fulfills two
tasks:
1) Load the experimental data from the raw data files stored in
   raw_data/ and stores it to viscortex_raw_data.json.
2) Process the data to derive complete sets of FLN, SLN, neuronal
   densities and laminar thicknesses and store these values to
   viscortex_processed_data.json.

All details of the procedures in this script are described in
Schmidt M, Bakker R, Hilgetag CC, Diesmann M & van Albada SJ
"Multi-scale account of the network structure of macaque visual cortex"
Brain Structure and Function (2018), 223:1409


Experimental Data
-----------------
Literature data consists of the following data that
will be stored in the corresponding dictionaries:

1. Hierarchy from Reid et al. (2009)
   ---> hierarchy
2. Layer-specific and overall neuronal densities from
   the lab of Helen Barbas
   ----> neuronal_density_data, neuronal_density_data_updated
3. Categorization of all areas, except MIP and MDP, into 8
   different structural classes
   ---> structure
4. Distances between areas with three different methods: Euclidean, Thom, and
   Median
   ---> euclidean_distances, thom_distances, median_distances
   The median distances are used for the multi-area model.
5. Surface areas from all areas
   ----> surfaces
6. CoCoMac data about the existence and patterns of connections
   between areas
   ----> cocomac
7. FLN data about extrinsic connections to three areas (V1,V2,V4)
   from Markov et al. (2011)
   ---> FLN_Data
8. FLN data about intrinsic connections of three areas (V1,V2,V4)
   from Markov et al. (2011)
   ---> intrinsic_FLN_Data
9. SLN data about connections to two areas (V1,V4) from
   Barone et al. (2000)
   ----> SLN_Data
10. Intrinsic connection probabilities from Potjans et al. (2012)
   ----> intrinsic_connectivity
11. Layer-specific number of neurons for cat V1 from Potjans et al.
    (2012) constructed from Binzegger et al. (2004)
    ----> Num_V1
12a. Thickness of layers from Beaulieu et al. (1983)
     ---> laminar_Thickness_cat
12b. Layer thicknesses for many areas collected from the literature
     ---> laminar_thicknesses
13. Total cortical thicknesses from Barbas lab
    ---> total_thickness_data
14. Translation from different schemes to FV91 scheme
15. Binzegger Data about relating synapse location to location of cell
    bodies
16. Average In-Degree from Cragg 1967 ---> AvInDegree


Authors
--------
Maximilian Schmidt
Sacha van Albada

"""
import numpy as np
import re
import copy
import json
import csv
import os
import pandas as pd
import subprocess

from itertools import product
from config import base_path
from nested_dict import nested_dict
from scipy import stats
from scipy import integrate


def process_raw_data():
    """
    Load and process raw data from literature.
    """

    """
    Helper variables and functions
    """
    area_list = ['V1', 'V2', 'VP', 'V3', 'PIP', 'V3A', 'MT', 'V4t', 'V4',
                 'PO', 'VOT', 'DP', 'MIP', 'MDP', 'MSTd', 'VIP', 'LIP',
                 'PITv', 'PITd', 'AITv', 'MSTl', 'FST', 'CITv', 'CITd',
                 '7a', 'STPp', 'STPa', 'FEF', '46', 'TF', 'TH', 'AITd']

    area_set = set(area_list)

    # to skip the explanatory headers in the .csv-files
    def skip_header():
        next(myreader)
        next(myreader)

    # to ignore the external sources in data
    def without_ext(l):
        s_l = set(l) & area_set
        return s_l

    """
    Set input and output paths
    """
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    datapath = os.path.join(basepath, 'raw_data')
    out_label = ''
    out_path = basepath

    """
    1. Define the hierarchy (continuous version from Reid et al. (2009))
    """
    hier_temp = pd.read_csv(os.path.join(datapath, 'hierarchy_Reid.csv'), skiprows=2,
                            sep='\t',
                            names=['area', 'level'])

    hierarchy = {area: level for area, level in hier_temp.values}
    for i in hierarchy.keys():
        if hierarchy[i] != '?':
            hierarchy[i] = float(hierarchy[i])
    hierarchy['MIP'] = .5
    hierarchy['MDP'] = .5

    # Hierarchy from Markov et al. 2014
    hier_temp = pd.read_csv(os.path.join(datapath, 'hierarchy_Markov.csv'),
                            sep=',',
                            skiprows=2,
                            names=['area', 'level', 'rescaled level'])

    hierarchy_markov = {}
    for i in range(len(hier_temp)):
        hierarchy_markov[hier_temp.iloc[i]['area']] = {'level': hier_temp.iloc[i][
            'level'], 'rescaled level': hier_temp.iloc[i]['rescaled level']}

    """
    2. Neuronal densities
    """
    # data obtained with NeuN staining
    # delivers total and laminar densities
    neuronal_density_data = {}
    temp = pd.read_csv(os.path.join(datapath, 'NeuronalDensities_NeuN.csv'), sep='\t',
                       skiprows=2,
                       names=['area', 'overall', 't_error', '23', '23_error',
                              '4', '4_error', '56', '56_error'])

    for i in np.arange(0, len(temp), 1):
        dict_ = {'overall': {'value': temp.iloc[i]['overall'],
                             'error': temp.iloc[i]['t_error']},
                 '23': {'value': temp.iloc[i]['23'],
                        'error': temp.iloc[i]['23_error']},
                 '4': {'value': temp.iloc[i]['4'],
                       'error': temp.iloc[i]['4_error']},
                 '56': {'value': temp.iloc[i]['56'],
                        'error': temp.iloc[i]['56_error']}}
        neuronal_density_data[temp.iloc[i]['area']] = dict_

    # data obtained with Nissl staining
    # delivers only total densities
    with open(os.path.join(datapath, 'NeuronalDensities_Nissl.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter=',')
        neuronal_density_data_updated = {}
        skip_header()
        for temp in myreader:
            try:
                if temp[0] == 'V5/MT':
                    neuronal_density_data_updated['MT'] = float(temp[2])
                if temp[0] == 'A46v':
                    neuronal_density_data_updated['area 46v'] = float(temp[2])
                if temp[0] == 'A7a':
                    neuronal_density_data_updated['7a'] = float(temp[2])
                if temp[0] == 'LIPv':
                    neuronal_density_data_updated['LIP'] = float(temp[2])
                if temp[0] == 'TEr':
                    neuronal_density_data_updated['Te1'] = float(temp[2])
                else:
                    neuronal_density_data_updated[temp[0]] = float(temp[2])
            except ValueError:
                pass

    """
    3. Architectural Types
    """
    temp = pd.read_csv(os.path.join(datapath, 'ArchitecturalTypes.csv'),
                       sep='\t',
                       skiprows=2,
                       names=['area', 'structure'])
    architecture = {area: architecture for area, architecture in temp.values}
    for i in architecture:
        if architecture[i] != '?':
            architecture[i] = int(architecture[i])

    """
    4. Distances
    """
    with open(os.path.join(datapath, 'Median_Distances_81areas.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        temp = next(myreader)
        areas = temp
        median_distance_data = {}
        for i in range(1, 82, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 82, 1):
                dict_[areas[j]] = float(temp[j])
            median_distance_data[areas[i]] = dict_

    with open(os.path.join(datapath, 'Thom_Distances.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        temp = next(myreader)
        areas = temp
        thom_distance_data = {}
        for i in range(1, 33, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 33, 1):
                dict_[areas[j]] = float(temp[j])
            thom_distance_data[areas[i]] = dict_

    # Distances for area in the parcellation used by Markov et al. (2014)
    with open(os.path.join(datapath, 'Thom_Distances_MERetal12.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        temp = next(myreader)
        areas = temp
        thom_distance_data_markov = {}
        for i in range(1, 93, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 93, 1):
                dict_[areas[j]] = float(temp[j])
            thom_distance_data_markov[areas[i]] = dict_

    with open(os.path.join(datapath, 'Euclidean_Distances.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        temp = next(myreader)
        areas = temp
        euclidean_distance_data = {}
        for i in range(1, 33, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 33, 1):
                dict_[areas[j]] = float(temp[j])
            euclidean_distance_data[areas[i]] = dict_

    """
    5. Surface areas
    """
    temp = pd.read_csv(os.path.join(datapath, 'cortical_surface.csv'),
                       sep='\t', skiprows=2,
                       names=['area', 'surface'])
    surface_data = {area: surface for area, surface in temp.values}

    """
    6. CoCoMac data
    """
    f = open(os.path.join(datapath, 'CoCoMac_complete_81.json'))
    dat = json.load(f)
    f.close()

    # Confirmative studies in cocomac
    f = open(os.path.join(datapath, 'cocomac_confirmative_studies.json'), 'r')
    confirmative = json.load(f)
    f.close()

    cocomac_conf = {}
    for target in area_list:
        cocomac_conf[target] = {}

    for source in confirmative:
        for target in confirmative[source]:
            if target.split('-')[-1] in area_list:
                cocomac_conf[target.split(
                    '-')[-1]][source.split('-')[-1]] = confirmative[source][target]

    # Negative studies in cocomac
    f = open(os.path.join(datapath, 'cocomac_negative_studies.json'), 'r')
    negative = json.load(f)
    f.close()

    cocomac_neg = {}
    for target in area_list:
        cocomac_neg[target] = {}

    for source in negative:
        for target in negative[source]:
            if target.split('-')[-1] in area_list:
                cocomac_neg[target.split(
                    '-')[-1]][source.split('-')[-1]] = negative[source][target]

    cocomac_data = {}
    for target in area_list:
        cocomac_data[target] = {}

    for source in dat:
        for target in dat[source]:
            source_pattern = dat[source][target][0]
            target_pattern = dat[source][target][1]

            if source_pattern is None and target_pattern is None:
                if confirmative[source][target] > 0 and target.split('-')[-1] in area_list:
                    cocomac_data[target.split('-')[-1]][source.split('-')[-1]] = {
                        'source_pattern': source_pattern, 'target_pattern': target_pattern}
            else:
                if source_pattern is not None:
                    source_pattern = list(source_pattern)
                if target_pattern is not None:
                    target_pattern = list(target_pattern)
                if target.split('-')[-1] in area_list:
                    cocomac_data[target.split('-')[-1]][source.split('-')[-1]] = {
                        'source_pattern': source_pattern, 'target_pattern': target_pattern}

    """
    7. FLN data
    """
    # FLN Data of Markov et al. 2012
    temp = pd.read_csv(os.path.join(datapath, 'Markov2014_FLN_rawdata.csv'),
                       sep='\t', skiprows=2,
                       usecols=['case', 'monkey', 'source_area', 'target_area',
                                'FLN', 'NLN', 'status'],
                       names=['case', 'monkey', 'source_area', 'target_area',
                              'FLN', 'NLN', 'status'])
    FLN_Data = {}
    for i in range(len(temp)):
        monkey = temp.iloc[i]['monkey']
        target = temp.iloc[i]['target_area']
        source = temp.iloc[i]['source_area']
        FLN = float(temp.iloc[i]['FLN'])

        status = temp.iloc[i]['status']

        if monkey in FLN_Data:
            FLN_Data[monkey]['source_areas'].update(
                {source: {'FLN': FLN, 'status': status}})
        else:
            FLN_Data[monkey] = {'target_area': target, 'source_areas': {
                source: {'FLN': FLN, 'status': status}}}

    NLN_Data = {}
    for i in range(0, len(temp)):
        monkey = temp.iloc[i]['monkey']
        target = temp.iloc[i]['target_area']
        source = temp.iloc[i]['source_area']
        NLN = float(temp.iloc[i]['NLN'])
        status = temp.iloc[i]['status']

        if monkey in NLN_Data:
            NLN_Data[monkey]['source_areas'].update(
                {source: {'NLN': NLN, 'status': status}})
        else:
            NLN_Data[monkey] = {'target_area': target, 'source_areas': {
                source: {'NLN': NLN, 'status': status}}}

    # Injection sites of Markov et al. (2014)
    temp = pd.read_csv(os.path.join(datapath, 'Markov2014_InjectionSites.csv'),
                       sep='\t', skiprows=3,
                       names=['injected area', 'case', 'monkey', 'section',
                              'plane', 'x', 'y', 'z', 'FV91 area'],
                       usecols=['injected area', 'case', 'monkey', 'section',
                                'plane', 'x', 'y', 'z', 'FV91 area'])

    injection_sites = {}
    for i in range(0, len(temp)):
        injection_sites[temp.iloc[i]['monkey']] = {'injected_area': temp.iloc[
            i]['injected area'], 'FV91_area': temp.iloc[i]['FV91 area']}

    """
    8. Intrinsic FLN_Data
    """
    with open(os.path.join(datapath, 'Intrinsic_FLN_Data.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        Intrinsic_FLN_Data = {}
        for i in range(4):
            temp = next(myreader)
            dict_ = {'mean': float(temp[1]), 'error': float(temp[2])}
            Intrinsic_FLN_Data[temp[0]] = dict_

    """
    9. SLN data
    """
    temp = pd.read_csv(os.path.join(datapath, 'SLN_Data.csv'),
                       skiprows=3,
                       sep=' ',
                       names=['index', 'target_area', 'source_area', 'S', 'I',
                              'TOT', 'DIST', 'DENS', 'monkey', 'lFLN', 'SLN',
                              'INJ', 'FLN', 'cSLN'])

    SLN_Data = {}

    for i in range(0, len(temp)):
        monkey = temp.iloc[i]['monkey']
        target = temp.iloc[i]['target_area']
        source = temp.iloc[i]['source_area']
        FLN = float(temp.iloc[i]['FLN'])
        S = temp.iloc[i]['S']
        I = temp.iloc[i]['I']
        TOT = temp.iloc[i]['TOT']
        SLN = temp.iloc[i]['SLN']

        if monkey in SLN_Data:
            SLN_Data[monkey]['source_areas'].update({source: {'FLN': float(FLN),
                                                              'S': int(S),
                                                              'I': int(I),
                                                              'TOT': int(TOT),
                                                              'SLN': float(SLN)}})
        else:
            SLN_Data[monkey] = {'target_area': target, 'source_areas':
                                {source: {'FLN': float(FLN),
                                          'S': int(S),
                                          'I': int(I),
                                          'TOT': int(TOT),
                                          'SLN': float(SLN)}}}

    """
    10. Intrinsic Connectivity from Potjans & Diesmann (2014)
    """
    with open(os.path.join(datapath, 'Intrinsic_Connectivity.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        intrinsic_connectivity = {}

        temp = next(myreader)

        areas = temp

        for i in range(1, 9, 1):
            temp = next(myreader)
            dict_ = {}
            for j in range(1, 10, 1):
                    dict_[areas[j]] = float(temp[j])
            intrinsic_connectivity[areas[i]] = dict_

    """
    11. Numbers of neurons and external inputs in V1
    """
    with open(os.path.join(datapath, 'Numbers_V1.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        num_V1 = {}
        for i in range(0, 9, 1):
            temp = next(myreader)
            num_V1[temp[0]] = {'neurons': float(
                temp[1]), 'ext_inputs': float(temp[2])}

    """
    Two alternatives for determining laminar thicknesses:
    """
    # 12a. Laminar thicknesses of cat area 17
    with open(os.path.join(datapath, 'Laminar_Thickness_cat.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        Laminar_Thickness_cat = {}

        for i in range(10):
            temp = next(myreader)
            Laminar_Thickness_cat[temp[0]] = {
                'thickness': float(temp[1]), 'error': float(temp[2])}

    # 12b. Laminar thicknesses of a large number of areas estimated from
    # micrographs from the literature
    with open(os.path.join(datapath, 'laminar_thicknesses_macaque.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        names = next(myreader)[1:16]
        for i in range(0, len(names)):
            names[i] = re.sub('L', '', names[i])
            names[i] = re.sub('/', '', names[i])
        laminar_thicknesses = {}
        line = True
        while line:
            try:
                temp = next(myreader)
            except StopIteration:
                line = False
            if temp[0] in laminar_thicknesses:
                if np.isscalar(laminar_thicknesses[temp[0]][names[0]]):
                    for j in range(len(temp) - 3):
                        if temp[j + 1]:
                            laminar_thicknesses[temp[0]][names[j]] = [
                                laminar_thicknesses[temp[0]][names[j]]] + [float(temp[j + 1])]
                        else:
                            laminar_thicknesses[temp[0]][names[j]] = [
                                laminar_thicknesses[temp[0]][names[j]]] + [np.nan]
                else:
                    for j in range(len(temp) - 3):
                        if temp[j + 1]:
                            laminar_thicknesses[temp[0]][names[j]] = laminar_thicknesses[
                                temp[0]][names[j]] + [float(temp[j + 1])]
                        else:
                            laminar_thicknesses[temp[0]][names[j]] = laminar_thicknesses[
                                temp[0]][names[j]] + [np.nan]
            else:
                laminar_thicknesses[temp[0]] = {}
                for j in range(len(temp) - 3):
                    if temp[j + 1]:
                        laminar_thicknesses[temp[0]
                                            ][names[j]] = float(temp[j + 1])
                    else:
                        laminar_thicknesses[temp[0]][names[j]] = np.nan

    """
    13. Total cortical thicknesses from Barbas lab
    """
    with open(os.path.join(datapath, 'CorticalThickness.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()
        next(myreader)
        total_thickness_data = {}
        for area in area_list:
            total_thickness_data[area] = np.nan

        for i in range(0, 30, 1):
            temp = next(myreader)
            if temp[4]:
                total_thickness_data[temp[0]] = float(
                    temp[4]) * 1000.  # convert to micrometer

    """
    14.Translation from Barbas' scheme to FV91 scheme
    """
    temp = pd.read_csv(os.path.join(datapath, 'SchemeTranslation.csv'),
                       sep='\t', skiprows=2,
                       names=['Markov_Area', 'FV91_area'])
    translation = {}
    for i, j in temp.values:
        if i in translation:
            translation[i] = np.append(translation[i], j)
        else:
            translation[i] = np.array([j])

    for i in translation:
        translation[i] = list(np.unique(translation[i]))

    f = open(os.path.join(datapath, 'overlap.json'), 'r')
    overlap = json.load(f)
    f.close()

    """
    15. Binzegger Data about relating synapse location to cell body location
    """
    with open(os.path.join(datapath, 'BinzeggerData.csv'), 'rt') as f:
        myreader = csv.reader(f, delimiter='\t')
        skip_header()

        pre_cells = next(myreader)
        binzegger_data = {}
        for i in range(1, 38, 1):
            temp = next(myreader)
            if temp[0] != '':
                subdict = {}
                if(temp[1]) == '':
                    syn_layer = re.sub("\D", "", re.sub(".*\(", "", temp[0]))
                else:
                    syn_layer = re.sub("\D", "", temp[1])
                for j in range(3, len(temp), 1):
                    try:
                        subdict[pre_cells[j]] = float(temp[j])
                    except:
                        subdict[pre_cells[j]] = temp[j]
                if temp[0] in binzegger_data:
                    binzegger_data[temp[0]]['syn_dict'].update(
                        {syn_layer: subdict})
                else:
                    binzegger_data[temp[0]] = {'occurrence': float(
                        temp[2]), 'syn_dict': {syn_layer: subdict}}

    """
    16. Average In-Degree
    """
    temp = np.loadtxt(os.path.join(datapath, 'SynapticDensity_Cragg.csv'),
                      delimiter='\t', skiprows=2)
    av_indegree_Cragg = float(temp)

    temp = np.loadtxt(os.path.join(datapath, 'SynapticDensity_OKusky.csv'),
                      delimiter='\t', skiprows=2)
    av_indegree_OKusky = float(temp)

    """
    Store experimental data in json file
    """
    raw_data = {'area_list': area_list,
                'hierarchy': hierarchy,
                'neuronal_density_data': neuronal_density_data,
                'architecture': architecture,
                'euclidean_distance_data': euclidean_distance_data,
                'thom_distance_data': thom_distance_data,
                'thom_distance_data_markov': thom_distance_data_markov,
                'median_distance_data': median_distance_data,
                'surface_data': surface_data,
                'cocomac_data': cocomac_data,
                'FLN_Data': FLN_Data,
                'Intrinsic_FLN_Data': Intrinsic_FLN_Data,
                'SLN_Data': SLN_Data,
                'NLN_Data': NLN_Data,
                'Intrinsic_Connectivity': intrinsic_connectivity,
                'num_V1': num_V1,
                'Laminar_Thickness_cat': Laminar_Thickness_cat,
                'laminar_thicknesses': laminar_thicknesses,
                'Binzegger_Data': binzegger_data,
                'av_indegree_Cragg': av_indegree_Cragg,
                'av_indegree_OKusky': av_indegree_OKusky,
                'hierarchy_markov': hierarchy_markov,
                'Translation': translation,
                'overlap': overlap,
                'total_thickness_data': total_thickness_data}

    with open(os.path.join(out_path,
                           ''.join(('viscortex_raw_data' + out_label + '.json'))),
              'w') as f:
        json.dump(raw_data, f)

    """
    Process experimental data
    =========================
    """
    # Assumption: MDP and MIP are on hierarchy levels 0.5 like
    # their neighboring areas PO,MT,V4t. With the same argument,
    # they are considered to be of structural type 5.
    hierarchy_completed = hierarchy.copy()
    architecture_completed = architecture.copy()

    hierarchy_completed['MIP'] = .5
    hierarchy_completed['MDP'] = .5
    architecture_completed['MIP'] = 5
    architecture_completed['MDP'] = 5

    """
    Neuronal numbers
    ----------------
    We compute neuronal numbers for each population
    by first deriving neuronal densities and laminar
    thicknesses and then combining them.

    ### Neuronal densities

    Transform neuronal density data to account for
    undersampling of cells by NeuN staining relative
    to Nissl staining.
    """

    # determine fit of scaling factors from NeuN to Nissl staining
    new = []
    old = []
    for area in neuronal_density_data_updated:
        if area in neuronal_density_data:
            old.append(neuronal_density_data[area]['overall']['value'])
            new.append(neuronal_density_data_updated[area])
    gradient, intercept, r_value, p_value, std_err = stats.linregress(old, new)

    # map neuronal density data to FV91 scheme
    neuronal_density_data_updated_FV91 = {}
    for i in list(neuronal_density_data_updated.keys()):
        if i not in area_list:
            if i in translation:
                areas_FV91 = translation[i]
                for kk in areas_FV91:
                    neuronal_density_data_updated_FV91[
                        kk] = neuronal_density_data_updated[i]
        else:
            neuronal_density_data_updated_FV91[
                i] = neuronal_density_data_updated[i]

    neuronal_density_data_FV91 = {}
    for i in list(neuronal_density_data.keys()):
        if i not in area_list:
            areas_FV91 = translation[i]
            for kk in areas_FV91:
                neuronal_density_data_FV91[kk] = neuronal_density_data[i].copy(
                )
        else:
            neuronal_density_data_FV91[i] = neuronal_density_data[i].copy()

    # map neuronal density data to 4 layers by dividing
    neuronal_density_data_FV91_4layers = {}
    for i in list(neuronal_density_data_FV91.keys()):
        neuronal_density_data_FV91_4layers.update({i: {'23': 0., 'overall': 0.,
                                                       '4': {'value': 0.0, 'error': 0.0},
                                                       '5': 0., '6': 0.}})

    for i in list(neuronal_density_data_FV91_4layers.keys()):
        for layer in ['23', '4', '56']:
            if neuronal_density_data_FV91[i][layer]['value'] > 0.:
                # Assign equal density to layers 5 and 6
                if layer == '56':
                    neuronal_density_data_FV91_4layers[i]['5'] = neuronal_density_data_FV91[
                        i]['56']['value'] * gradient + intercept
                    neuronal_density_data_FV91_4layers[i]['6'] = neuronal_density_data_FV91[
                        i]['56']['value'] * gradient + intercept
                else:
                    neuronal_density_data_FV91_4layers[i][layer] = neuronal_density_data_FV91[
                        i][layer]['value'] * gradient + intercept
            else:
                neuronal_density_data_FV91_4layers[i][layer] = 0.
        # if there is Nissl data, then take it, otherwise
        # transform NeuN data with the linear fit
        if i in neuronal_density_data_updated_FV91:
            neuronal_density_data_FV91_4layers[i][
                'overall'] = neuronal_density_data_updated_FV91[i]
        else:
            neuronal_density_data_FV91_4layers[i]['overall'] = neuronal_density_data_FV91[
                i]['overall']['value'] * gradient + intercept
    """
    We build a dictionary containing neuron densities
    (overall and layer-specific) for each area. If there
    is no data available for an area, we compute the
    densities in two steps:

    1. Assign an average neural density to each cortical
       category for each layer and overall density.
    2. Based on the category, assign a density to each
       area, if there is no direct data available.

    In contrast to the model, the experimental data
    combines layers 5 and 6 to one layer. Thus, we
    assign values to 5 and 6 separately by the following calculation:
    N56 = N5 + N6, d56 = d5 + d6, A56 = A5 = A6
    rho56 = N56 / (A56*d56) = (N5+N6) / (A56*(d5+d6)) = N5/(A56*(d5+d6)) +
    N6/(A56*(d5+d6)) = N5/(A5*d5) * d5/(d5+d6) + N6/(A6*d6) * d6/(d5+d6) =
    rho5 * d5/(d5+d6) + rho6 * d6/(d5+d6) = rho5 * factor + rho6 *
    (1-factor)

    """

    # Step 1: Assign an average density to each structural type
    neuronal_density_list = [{'overall': [], '23': [], '4': [], '5': [],
                              '6': []} for i in range(8)]

    for area in list(neuronal_density_data_FV91_4layers.keys()):
        category = architecture_completed[area]
        for key in list(neuronal_density_data_FV91_4layers[area].keys()):
            neuronal_density_list[category - 1][key].append(float(
                neuronal_density_data_FV91_4layers[area][key]))

    category_density = {}

    for x in range(8):
        dict_ = {}
        for i in list(neuronal_density_list[0].keys()):
            if len(neuronal_density_list[x][i]) == 0:
                dict_[i] = np.nan
            else:
                dict_[i] = np.mean(neuronal_density_list[x][i])
            category_density[x + 1] = dict_

    # Step 2: For areas with out experimental data,
    # assign the average density values of their structural type
    neuronal_densities = nested_dict()
    for area in list(architecture_completed.keys()):
        dict_ = {}
        if architecture_completed[area] in range(1, 9, 1):
            if area in list(neuronal_density_data_FV91_4layers.keys()):
                for key in list(neuronal_density_data_FV91_4layers[area].keys()):
                    neuronal_densities[area][key] = neuronal_density_data_FV91_4layers[
                        area][key]
            else:
                neuronal_densities[area] = category_density[architecture_completed[area]]
        else:
            neuronal_densities[area] = '?'
    neuronal_densities = neuronal_densities.to_dict()

    """
    ### Thicknesses

    To convert the neuronal volume densities into
    neuron counts, we need total and laminar thicknesses
    for each area.

    For areas without experimental data on thicknesses, we
    use we use a linear fit of total thickness vs.
    logarithmic overall neuron density.

    In addition, we use linear fits of relative thicknesses
    vs. logarithmic neuron densities for L4 thickness, and the
    arithmetic mean of the data for L1, L2/3, L5 and L6.

    Finally, we convert the relative laminar thicknesses
    into absolute values by multiplying with the total thickness.



    Total thicknesses
    """
    # linear regression of barbas thicknesses vs architectural types
    barbas_array = np.zeros(len(area_list))
    log_density_array = np.zeros(len(area_list))
    for i, area in enumerate(area_list):
        barbas_array[i] = total_thickness_data[area]
        log_density_array[i] = np.log10(neuronal_densities[area]['overall'])

    gradient, intercept, r_value, p_value, std_err = stats.linregress(
        log_density_array[np.isfinite(barbas_array)], barbas_array[np.isfinite(barbas_array)])

    # total thicknesses
    total_thicknesses = total_thickness_data.copy()
    for a in list(total_thicknesses.keys()):
        if np.isnan(total_thicknesses[a]):
            total_thicknesses[a] = intercept + gradient * \
                np.log10(neuronal_densities[a]['overall'])

    """
    Laminar thicknesses
    """
    # calculate relative layer thicknesses for each area and study
    frac_of_total = nested_dict()
    for area in list(laminar_thicknesses.keys()):
        for layer in list(laminar_thicknesses[area].keys()):
            frac_of_total[area][layer] = np.array(laminar_thicknesses[area][
                layer]) / np.array(laminar_thicknesses[area]['total'])
            # if layer thickness is zero, it makes up 0% of the total, even if the
            # total is unknown
            if 0 in np.array(laminar_thicknesses[area][layer]):
                if np.isscalar(laminar_thicknesses[area][layer]):
                    frac_of_total[area][layer] = 0
                else:
                    indices = np.where(
                        np.array(laminar_thicknesses[area][layer]) == 0)[0]
                    for i in indices:
                        frac_of_total[area][layer][i] = 0
    frac_of_total = frac_of_total.to_dict()

    # for areas for which these are known: mean across studies
    # of fractions of total thickness occupied by each layer
    relative_layer_thicknesses = nested_dict()
    for area, layer in product(area_list, ['1', '23', '4', '5', '6']):
        if np.isscalar(frac_of_total[area][layer]):
            relative_layer_thicknesses[area][layer] = frac_of_total[area][layer]
        else:
            relative_layer_thicknesses[area][layer] = np.mean(
                frac_of_total[area][layer][np.isfinite(frac_of_total[area][layer])])
    relative_layer_thicknesses = relative_layer_thicknesses.to_dict()

    # for areas where these data are missing, use mean across areas of
    # fractions of total thickness occupied by L1, L2/3, by L5, and by L6
    tmp1 = np.array([])
    tmp23 = np.array([])
    tmp5 = np.array([])
    tmp6 = np.array([])
    for area in list(relative_layer_thicknesses.keys()):
        tmp1 = np.append(tmp1, relative_layer_thicknesses[area]['1'])
        tmp23 = np.append(tmp23, relative_layer_thicknesses[area]['23'])
        tmp5 = np.append(tmp5, relative_layer_thicknesses[area]['5'])
        tmp6 = np.append(tmp6, relative_layer_thicknesses[area]['6'])

    mean_rel_L1_thickness = np.mean(tmp1[np.isfinite(tmp1)])
    mean_rel_L23_thickness = np.mean(tmp23[np.isfinite(tmp23)])
    mean_rel_L5_thickness = np.mean(tmp5[np.isfinite(tmp5)])
    mean_rel_L6_thickness = np.mean(tmp6[np.isfinite(tmp6)])

    for area in list(relative_layer_thicknesses.keys()):
        if np.isnan(relative_layer_thicknesses[area]['1']):
            relative_layer_thicknesses[area]['1'] = mean_rel_L1_thickness
        if np.isnan(relative_layer_thicknesses[area]['23']):
            relative_layer_thicknesses[area]['23'] = mean_rel_L23_thickness
        if np.isnan(relative_layer_thicknesses[area]['5']):
            relative_layer_thicknesses[area]['5'] = mean_rel_L5_thickness
        if np.isnan(relative_layer_thicknesses[area]['6']):
            relative_layer_thicknesses[area]['6'] = mean_rel_L6_thickness

    # mean relative laminar thickness across studies for each area
    frac4_of_total = np.zeros(len(area_list))

    for i, area in enumerate(area_list):
        temp = frac_of_total[area]['4']
        if not np.isscalar(temp):
            if sum(np.isfinite(temp)):
                frac4_of_total[i] = np.nansum(temp) / sum(np.isfinite(temp))
            else:
                frac4_of_total[i] = np.nan
        else:
            frac4_of_total[i] = temp

    # perform regressions of per-area average relative
    # laminar thicknesses vs logarithmic overall densities
    gradient4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(np.array(
        log_density_array)[np.isfinite(frac4_of_total)], frac4_of_total[
            np.isfinite(frac4_of_total)])

    # assign values based on linear regressions
    for area in list(relative_layer_thicknesses.keys()):
        if np.isnan(relative_layer_thicknesses[area]['4']):
            relative_layer_thicknesses[area]['4'] = intercept4 + gradient4 * np.log10(
                neuronal_densities[area]['overall'])

    # convert relative thicknesses into absolute ones
    laminar_thicknesses_completed = {}
    for area in list(relative_layer_thicknesses.keys()):
        laminar_thicknesses_completed[area] = {}
        sum_rel_thick = sum(relative_layer_thicknesses[area].values())
        for layer in list(relative_layer_thicknesses[area].keys()):
            # 0.001 converts from micrometer to mm; the result is normalized to
            # have the sum of the relative thicknesses equal to 1
            laminar_thicknesses_completed[area][layer] = 0.001 * relative_layer_thicknesses[
                area][layer] * total_thicknesses[area] / sum_rel_thick

    """
    Finally, we compute neuron numbers for each population.
    We assume a laminar-specific ratio of excitatory
    to inhibitory neurons to be constant across areas.
    """
    EI_ratio = {'23': num_V1['23E']['neurons'] / (
        num_V1['23E']['neurons'] + num_V1['23I']['neurons']),
                '4': num_V1['4E']['neurons'] / (num_V1['4E']['neurons'] +
                                                num_V1['4I']['neurons']),
                '5': num_V1['5E']['neurons'] / (num_V1['5E']['neurons'] +
                                                num_V1['5I']['neurons']),
                '6': num_V1['6E']['neurons'] / (num_V1['6E']['neurons'] +
                                                num_V1['6I']['neurons'])}

    """
    Then, we compute the number of neurons in
    population i in layer v_i of area A as
    N(A,i) = rho(A,v_i) * S(A) * D(A,v_i) * EI_ratio.
    """
    realistic_neuronal_numbers = nested_dict()
    for area in list(neuronal_densities.keys()):
        S = surface_data[area]
        ltc = laminar_thicknesses_completed[area]
        nd = neuronal_densities[area]
        realistic_neuronal_numbers[area]['23E'] = (S * ltc['23'] *
                                                   nd['23'] * EI_ratio['23'])
        realistic_neuronal_numbers[area]['23I'] = (realistic_neuronal_numbers[area]['23E'] *
                                                   (1. - EI_ratio['23']) / EI_ratio['23'])
        realistic_neuronal_numbers[area]['4E'] = (S * ltc['4'] *
                                                  nd['4'] * EI_ratio['4'])
        realistic_neuronal_numbers[area]['4I'] = (realistic_neuronal_numbers[area]['4E'] *
                                                  (1. - EI_ratio['4']) / EI_ratio['4'])
        realistic_neuronal_numbers[area]['5E'] = (S * ltc['5'] *
                                                  nd['5'] * EI_ratio['5'])
        realistic_neuronal_numbers[area]['5I'] = (realistic_neuronal_numbers[area]['5E'] *
                                                  (1. - EI_ratio['5']) / EI_ratio['5'])
        realistic_neuronal_numbers[area]['6E'] = (S * ltc['6'] *
                                                  nd['6'] * EI_ratio['6'])
        realistic_neuronal_numbers[area]['6I'] = (realistic_neuronal_numbers[area]['6E'] *
                                                  (1. - EI_ratio['6']) / EI_ratio['6'])
        realistic_neuronal_numbers[area]['total'] = sum(
            realistic_neuronal_numbers[area].values())
    realistic_neuronal_numbers = realistic_neuronal_numbers.to_dict()

    """
    Cortico-cortical connectivity
    -----------------------------

    ### FLN values

    We proceed with the FLN values in three steps:
    1. Map the injection sites (= target areas) to the
       FV91 scheme.
    2. Map the source areas to FV91 with the overlap tool.
    3. Retrieve information about existing connections
       from CoCoMac.
    4. Fit exponential distance rule to existing data.
    5. Fill missing values with exponential distance rule.
    """

    FLN_Data_FV91 = {}
    """
    1. Map target areas to FV91 according to the
       exact coordinates of injected areas_FV91.
    """

    for monkey in FLN_Data:
        FV91_area = injection_sites[monkey]['FV91_area']
        if FV91_area in area_list:
            if FV91_area in FLN_Data_FV91:
                for source in FLN_Data[monkey]['source_areas']:
                    if source in FLN_Data_FV91[FV91_area]:
                        FLN_Data_FV91[FV91_area][source] = np.append(FLN_Data_FV91[
                            FV91_area][source],
                                                                     FLN_Data[monkey][
                                                                         'source_areas'][
                                                                             source]['FLN'])
                    else:
                        FLN_Data_FV91[FV91_area][source] = np.array(
                            [FLN_Data[monkey]['source_areas'][source]['FLN']])
            else:
                FLN_Data_FV91[FV91_area] = {}
                for source in FLN_Data[monkey]['source_areas']:
                    FLN_Data_FV91[FV91_area][source] = np.array(
                        [FLN_Data[monkey]['source_areas'][source]['FLN']])

    # Compute the arithmetic means and fill arrays with zeros if necessary:
    for target in FLN_Data_FV91:
        dimension = 0
        for source in FLN_Data_FV91[target]:
            if FLN_Data_FV91[target][source].size > dimension:
                dimension = FLN_Data_FV91[target][source].size
        for source in FLN_Data_FV91[target]:
            array = np.append(FLN_Data_FV91[target][source], np.zeros(
                dimension - FLN_Data_FV91[target][source].size))
            FLN_Data_FV91[target][source] = np.mean(array)

    """
    2. Map the source areas according to the overlap tool
    """
    FLN_Data_FV91_mapped = {}
    for target in FLN_Data_FV91:
        FLN_Data_FV91_mapped[target] = {}

        for source in FLN_Data_FV91[target]:
            # Here, we have to translate some of the area names.
            source_key = source + '_M132'
            if source == 'ENTO':
                source_key = 'Entorhinal' + '_M132'
            if source == 'POLE':
                source_key = 'Temporal-pole' + '_M132'
            if source == 'Parainsula':
                source_key = 'Insula' + '_M132'
            if source == 'SUB':
                source_key = 'Subiculum' + '_M132'
            if source == 'PIRI':
                source_key = 'Piriform' + '_M132'
            if source == 'PERI':
                source_key = 'Perirhinal' + '_M132'
            if source == 'Pro.St.':
                source_key = 'Prostriate' + '_M132'
            if source == 'INSULA':
                source_key = 'Insula' + '_M132'
            if source == 'CORE':
                source_key = 'Aud-core' + '_M132'
            if source == '29/30':
                source_key = '29_30' + '_M132'
            if source == 'TEa/mp':
                source_key = 'TEam-p' + '_M132'
            if source == 'TH/TF':
                source_key = 'TH_TF' + '_M132'
            if source == 'TEa/ma':
                source_key = 'TEam-a' + '_M132'
            if source == '9/46v':
                source_key = '9_46v' + '_M132'
            if source == '9/46d':
                source_key = '9_46d' + '_M132'
            if source == 'SII':
                source_key = 'S2' + '_M132'

            for FV91_key in overlap['all'][source_key]:
                FV91_source = re.sub(
                    "FVE.", "", re.sub("FVE_all.", "", FV91_key))
                if FV91_source in area_list:
                    if FV91_source in FLN_Data_FV91_mapped[target]:
                        FLN_Data_FV91_mapped[target][FV91_source] += overlap['all'][
                            source_key][FV91_key] / 100. * FLN_Data_FV91[target][source]
                    else:
                        FLN_Data_FV91_mapped[target][FV91_source] = overlap['all'][
                            source_key][FV91_key] / 100. * FLN_Data_FV91[target][source]

    # Remove intrinsic FLN from FLN_Data_FV91_mapped
    for target in FLN_Data_FV91_mapped:
        if target in FLN_Data_FV91_mapped[target]:
            del FLN_Data_FV91_mapped[target][target]

    # Fill up FLN_Data_FV91 with missing target areas
    for a in area_list:
        if a not in FLN_Data_FV91:
            FLN_Data_FV91_mapped.update({a: {}})

    """
    3. Process CoCoMac information
       In the laminar patterns, make replacements:
       'X' --> 2
       '?' --> 0 or -1
    """
    cocomac_completed = {}
    for target in cocomac_data:
        cocomac_completed[target] = {}
        for source in cocomac_data[target]:
            sp = cocomac_data[target][source]['source_pattern']
            tp = cocomac_data[target][source]['target_pattern']
            if sp is not None:
                for i in range(6):
                    if sp[i] == 'X':
                        sp[i] = 2
                    if sp[i] == '?' and i in [0, 3]:
                        sp[i] = 0
                    if sp[i] == '?' and i in [1, 2, 4, 5]:
                        # Dummy value to enable transformation of this list into a
                        # numpy array
                        sp[i] = -1

            if tp is not None:
                for i in range(6):
                    if tp[i] == 'X':
                        tp[i] = 2.
                    if tp[i] == '?':
                        # Dummy value to enable transformation of this list into a
                        # numpy array
                        tp[i] = -1

            cocomac_completed[target][source] = {
                'source_pattern': sp, 'target_pattern': tp}

    # Add newly found connections by Markov et al. to cocomac data
    for target_area in list(cocomac_data.keys()):
        coco_source = list(cocomac_data[target_area].keys())
        for source_area in without_ext(list(FLN_Data_FV91_mapped[target_area].keys())):
            if (source_area not in coco_source and source_area != target_area and
                    FLN_Data_FV91_mapped[target_area][source_area] > 0.):
                cocomac_completed[target_area][source_area] = {
                    'target_pattern': None, 'source_pattern': None}

    # add self-connections to cocomac_completed for consistency reasons
    for area in area_list:
        if area not in cocomac_completed[area]:
            cocomac_completed[area][area] = {
                'target_pattern': None, 'source_pattern': None}

    """
    4. Fill missing data with fitted values from
       exponential distance rule.
    """
    FLN_values_FV91 = np.array([])
    distances_FV91 = np.array([])

    for target_area in FLN_Data_FV91_mapped:
        for source_area in FLN_Data_FV91_mapped[target_area]:
            if target_area in thom_distance_data and source_area in thom_distance_data:
                if FLN_Data_FV91_mapped[target_area][source_area]:
                    FLN_values_FV91 = np.append(FLN_values_FV91, FLN_Data_FV91_mapped[
                                                target_area][source_area])
                    distances_FV91 = np.append(distances_FV91, median_distance_data[
                                               target_area][source_area])

    # Linear Fit to log values
    gradient, intercept, r_value, p_value, std_err = stats.linregress(
        distances_FV91, np.log(FLN_values_FV91))
    EDR_params = [intercept, gradient]

    def EDR(target_area, source_area):
        return np.exp(EDR_params[0] + EDR_params[1] *
                      median_distance_data[target_area][source_area])

    """
    5. Build dictionary with FLN value for each
       connection of the model.
    """
    FLN_completed = {}
    for target_area in cocomac_completed:
        # We have data for some of the connections to target_area.
        if len(list(FLN_Data_FV91_mapped[target_area].keys())) > 0:
            FLN_completed[target_area] = copy.deepcopy(
                FLN_Data_FV91_mapped[target_area])
            for source_area in cocomac_completed[target_area]:
                if (source_area not in FLN_Data_FV91_mapped[target_area] and
                    source_area in median_distance_data and
                        source_area != target_area):
                    FLN_completed[target_area][
                        source_area] = EDR(target_area, source_area)
        # We have no data for any connection to target_area.
        else:
            FLN_completed[target_area] = {}
            for source_area in cocomac_completed[target_area]:
                if source_area in median_distance_data:
                    if source_area != target_area:
                        FLN_completed[target_area][
                            source_area] = EDR(target_area, source_area)

    # Assign all FLN values from non-visual areas to "external FLN"
    for target_area in FLN_completed:
        ext_FLN = 0.
        sources = list(FLN_completed[target_area].keys())
        for source_area in sources:
            if source_area not in area_list:
                ext_FLN += FLN_completed[target_area][source_area]
                del FLN_completed[target_area][source_area]
        FLN_completed[target_area]['external'] = ext_FLN

    """
    ### SLN values

    1. Map target areas the exact coordinates
       of injections and compute the arithmetic mean
       over injections.
    2. Map the source areas using the overlap tool.
    3. Perform sigmoidal fit of SLN values vs.
       logarithmic neuron density ratios in R.
    4. Fill missing data with fitted values.




    1. Map target areas
    """
    SLN_Data_FV91 = {}
    for monkey in SLN_Data:
        FV91_area = injection_sites[monkey]['FV91_area']

        if FV91_area in area_list:
            if FV91_area in SLN_Data_FV91:
                for source in SLN_Data[monkey]['source_areas']:
                    if source in SLN_Data_FV91[FV91_area]:
                        SLN_Data_FV91[FV91_area][source] = np.append(SLN_Data_FV91[
                            FV91_area][source],
                                                                     SLN_Data[
                                                                         monkey]['source_areas'][
                                                                         source]['SLN'])
                    else:
                        SLN_Data_FV91[FV91_area][source] = np.array(
                            [SLN_Data[monkey]['source_areas'][source]['SLN']])
            else:
                SLN_Data_FV91[FV91_area] = {}
                for source in SLN_Data[monkey]['source_areas']:
                    SLN_Data_FV91[FV91_area][source] = np.array(
                        [SLN_Data[monkey]['source_areas'][source]['SLN']])

    for target in SLN_Data_FV91:
        dimension = 0
        for source in SLN_Data_FV91[target]:
            if SLN_Data_FV91[target][source].size > dimension:
                dimension = SLN_Data_FV91[target][source].size
        for source in SLN_Data_FV91[target]:
            array = np.append(SLN_Data_FV91[target][source], np.zeros(
                dimension - SLN_Data_FV91[target][source].size))
            SLN_Data_FV91[target][source] = np.mean(array)

    """
    2. Map source areas
       To map the data from M132 to FV91, we weight
       the SLN by the overlap between the source area
       in the former and the source area in the latter
       scheme and the FLN to take into account the overall
       strength of the connection.
    """
    SLN_Data_FV91_mapped = {}
    for target in SLN_Data_FV91:
        SLN_Data_FV91_mapped[target] = {}

        for source in SLN_Data_FV91[target]:

            source_key = source + '_M132'
            if source == 'ENTO':
                source_key = 'Entorhinal' + '_M132'
            if source == 'POLE':
                source_key = 'Temporal-pole' + '_M132'
            if source == 'Parainsula':
                source_key = 'Insula' + '_M132'
            if source == 'SUB':
                source_key = 'Subiculum' + '_M132'
            if source == 'PIRI':
                source_key = 'Piriform' + '_M132'
            if source == 'PERI':
                source_key = 'Perirhinal' + '_M132'
            if source == 'Pro.St.':
                source_key = 'Prostriate' + '_M132'
            if source == 'INSULA':
                source_key = 'Insula' + '_M132'
            if source == 'CORE':
                source_key = 'Aud-core' + '_M132'
            if source == '29/30':
                source_key = '29_30' + '_M132'
            if source == 'TEa/mp':
                source_key = 'TEam-p' + '_M132'
            if source == 'TH/TF':
                source_key = 'TH_TF' + '_M132'
            if source == 'TEa/ma':
                source_key = 'TEam-a' + '_M132'
            if source == '9/46v':
                source_key = '9_46v' + '_M132'
            if source == '9/46d':
                source_key = '9_46d' + '_M132'
            if source == 'SII':
                source_key = 'S2' + '_M132'

            for FV91_key in overlap['all'][source_key]:
                FV91_source = re.sub(
                    "FVE.", "", re.sub("FVE_all.", "", FV91_key))
                if FV91_source in area_list:
                    if FV91_source in SLN_Data_FV91_mapped[target]:
                        SLN_Data_FV91_mapped[target][FV91_source]['S'] += (overlap['all'][
                            source_key][FV91_key] / 100. * SLN_Data_FV91[
                                target][source] * FLN_Data_FV91[target][source])

                        SLN_Data_FV91_mapped[target][FV91_source]['I'] += overlap['all'][
                            source_key][FV91_key] / 100. * (1. - SLN_Data_FV91[
                                target][source]) * FLN_Data_FV91[target][source]
                    else:
                        SLN_Data_FV91_mapped[target][FV91_source] = {}
                        SLN_Data_FV91_mapped[target][FV91_source]['S'] = overlap['all'][source_key][
                            FV91_key] / 100. * SLN_Data_FV91[
                                target][source] * FLN_Data_FV91[target][source]
                        SLN_Data_FV91_mapped[target][FV91_source]['I'] = overlap['all'][source_key][
                            FV91_key] / 100. * (1. - SLN_Data_FV91[
                                target][source]) * FLN_Data_FV91[target][source]

    for target in SLN_Data_FV91_mapped:
        for source in SLN_Data_FV91_mapped[target]:
            SLN_Data_FV91_mapped[target][source] = SLN_Data_FV91_mapped[target][source][
                'S'] / (SLN_Data_FV91_mapped[target][source]['S'] +
                        SLN_Data_FV91_mapped[target][source]['I'])

    """
    3. Sigmoidal fit of SLN vs. logarithmic ratio of neuron densities.
    """

    def integrand(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    def probit(x,):
        if isinstance(x, np.ndarray):
            res = [integrate.quad(integrand, -1000., i,
                                  args=(0., 1.))[0] for i in x]
        else:
            res = integrate.quad(integrand, -1000., x, args=(0., 1.))[0]
        return res

    # Call R script to perform SLN fit
    try:
        proc = subprocess.Popen(["Rscript",
                                 os.path.join(basepath, 'SLN_logdensities.R'),
                                 base_path],
                                stdout=subprocess.PIPE)
        out = proc.communicate()[0].decode('utf-8')
        R_fit = [float(out.split('\n')[1].split(' ')[1]),
                     float(out.split('\n')[1].split(' ')[3])]
    except OSError:
        print("No R installation, taking hard-coded SLN fit parameters.")
        R_fit = [-0.1516142, -1.5343200]

    """
    4. Fill missing data with fitted values.
    """
    SLN_completed = {}
    s = 0.
    s2 = 0.
    for target in area_list:
        SLN_completed[target] = {}
        for source in list(cocomac_completed[target].keys()):
            if source in area_list and source != target:
                if target in SLN_Data_FV91_mapped and source in SLN_Data_FV91_mapped[target]:
                    value = SLN_Data_FV91_mapped[target][source]
                    s += 1
                else:
                    nd_target = neuronal_densities[target]
                    nd_source = neuronal_densities[source]
                    x = R_fit[1] * float(np.log(nd_target['overall']) -
                                         np.log(nd_source['overall'])) + R_fit[0]
                    value = probit(x)
                    s2 += 1
                SLN_completed[target][source] = value

    """
    Write output files
    ------------------

    Store processed values to json file.
    """
    processed_data = {'cocomac_completed': cocomac_completed,
                      'architecture_completed': architecture_completed,
                      'hierarchy_completed': hierarchy_completed,
                      'SLN_completed': SLN_completed,
                      'SLN_Data_FV91': SLN_Data_FV91_mapped,
                      'FLN_Data_FV91': FLN_Data_FV91_mapped,
                      'FLN_completed': FLN_completed,
                      'neuronal_densities': neuronal_densities,
                      'neuronal_density_data_FV91_4layers': neuronal_density_data_FV91_4layers,
                      'realistic_neuronal_numbers': realistic_neuronal_numbers,
                      'total_thicknesses': total_thicknesses,
                      'laminar_thicknesses': laminar_thicknesses_completed,
                      'category_density': category_density
                      }

    with open(os.path.join(out_path,
                           ''.join(('viscortex_processed_data', out_label, '.json'))),
              'w') as f:
        json.dump(processed_data, f)


if __name__ == '__main__':
    process_raw_data()
