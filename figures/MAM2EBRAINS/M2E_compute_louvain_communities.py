# import community
from community import community_louvain
import csv
import json
import networkx as nx
import numpy as np
import os

def compute_communities(M, data_path, label):
    """
    Determines communities in the functional connectivity of either the
    experimental fMRI data used in Schmidt et al. 2018 or of a given
    simulation (the functional connectivity being based either on spike
    rates or an estimated BOLD signal).

    Parameters:
        - M (MultiAreaModel): The M object containing the area list.
        - data_path (str): The path to the data directory.
        - label (str): The label for the data.

    Returns:
        None
    """
    method = "synaptic_input"
    
    if label == 'exp':
        load_path = ''

        func_conn_data = {}
        
        with open('./figures/Schmidt2018_dyn/Fig8_exp_func_conn.csv', 'r') as f:
            myreader = csv.reader(f, delimiter='\t')
            # Skip first 3 lines
            next(myreader)
            next(myreader)
            next(myreader)
            areas = next(myreader)
            for line in myreader:
                dict_ = {}
                for i in range(len(line)):
                    dict_[areas[i]] = float(line[i])
                func_conn_data[areas[myreader.line_num - 5]] = dict_

        FC = np.zeros((len(M.area_list),
                       len(M.area_list)))
        for i, area1 in enumerate(M.area_list):
            for j, area2 in enumerate(M.area_list):
                FC[i][j] = func_conn_data[area1][area2]

    else:
        load_path = os.path.join(data_path,
                                 label,
                                 'Analysis',
                                 'functional_connectivity_{}.npy'.format(method))
        FC = np.load(load_path)

    # Set diagonal to 0
    for i in range(FC.shape[0]):
        FC[i][i] = 0.

    G = nx.Graph()
    for area in M.area_list:
        G.add_node(area)

    edges = []
    for i, area in enumerate(M.area_list):
        for j, area2 in enumerate(M.area_list):
            edges.append((area, area2, FC[i][j]))
    G.add_weighted_edges_from(edges)

    part = community_louvain.best_partition(G)

    if label == 'exp':
        fn = os.path.join('FC_exp_communities.json')
    else:
        fn = os.path.join(data_path,
                          label,
                          'Analysis',
                          'FC_{}_communities.json'.format(method))

    with open(fn, 'w') as f:
        json.dump(part, f)
