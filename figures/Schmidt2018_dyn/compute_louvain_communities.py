import community
import json
import networkx as nx
import numpy as np
import os
import sys

from multiarea_model.multiarea_model import MultiAreaModel

data_path = sys.argv[1]
label = sys.argv[2]


"""
Create MultiAreaModel instance to have access to data structures
"""
M = MultiAreaModel({})


load_path = os.path.join(data_path,
                         label,
                         'Analysis',
                         'functional_connectivity_synaptic_input.npy')


FC = np.load(load_path)
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

fn = os.path.join(data_path,
                  label,
                  'Analysis',
                  'FC_synaptic_input_communities.json')

with open(fn, 'w') as f:
    json.dump(part, f)
