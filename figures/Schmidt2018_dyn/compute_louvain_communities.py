import community
import csv
import json
import networkx as nx
import numpy as np
import os
import sys

from multiarea_model.multiarea_model import MultiAreaModel

"""
Determines communities in the functional connectivity of either the
experimental fMRI data used in Schmidt et al. 2018 or of a given
simulation (the functional connectivity being based either on spike
rates or an estimated BOLD signal).
"""

data_path = sys.argv[1]
label = sys.argv[2]
method = sys.argv[3]

"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})

if label == 'exp':
    load_path = ''

    func_conn_data = {}
    with open('Fig8_exp_func_conn.csv', 'r') as f:
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

part = community.best_partition(G)

if label == 'exp':
    fn = os.path.join('FC_exp_communities.json')
else:
    fn = os.path.join(data_path,
                      label,
                      'Analysis',
                      'FC_{}_communities.json'.format(method))

with open(fn, 'w') as f:
    json.dump(part, f)
