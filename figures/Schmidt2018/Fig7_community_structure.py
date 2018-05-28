import os
import copy
from helpers import area_list, infomap_path
import numpy as np
from config import base_path
from graph_helpers import apply_map_equation, modularity
from graph_helpers import create_graph, plot_clustered_graph
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import load_degree_data
from plotcolors import myred_hex, myblue_hex, myred2_hex
from plotcolors import myblue2_hex, mypurple_hex, myyellow_hex, mygreen_hex

# NOTE: igraph does not support setting a random seed for random
# placing of graph nodes, so the resulting plot will look slightly
# different from the published version

colors = [myred_hex, myblue_hex, myblue2_hex, mygreen_hex,
          mypurple_hex, myyellow_hex, myred2_hex, '#888888']
colors = [myred_hex, myblue_hex, myblue2_hex,
          mypurple_hex, mygreen_hex, myred2_hex]
base_dir = os.getcwd()

# Define positions of clusters
center_of_masses = [[2., 1.],
                    [-2., 1.],
                    [0., 1.],
                    [0., 0.],
                    [0., 2.],
                    [0.9, 1.]]

"""
Initialize model instance and load connectivity data
"""
M = MultiAreaModel({})
fn = os.path.join(base_path,
                  'config_files',
                  'custom_Data_Model_{}.json'.format(M.label))
ind, inda, out, outa = load_degree_data(fn)

"""
Construct matrix of relative and absolute outdegrees
"""
conn_matrix = np.zeros((32, 32))
conn_matrix_abs = np.zeros((32, 32))
for i, area1 in enumerate(area_list):
    for j, area2 in enumerate(area_list):
        value = outa[area1][area2] / np.sum(list(outa[area1].values()))
        value_abs = outa[area1][area2]
        if area1 != area2:
            conn_matrix[i][j] = value
            conn_matrix_abs[i][j] = value_abs


"""
Create igraph.Graph instances
with relative and absolute outdegrees.
"""
g = create_graph(conn_matrix, area_list)
g_abs = create_graph(conn_matrix_abs, area_list)


"""
Determine clusters using the map equation.
"""
modules, modules_areas, index = apply_map_equation(
    conn_matrix, area_list, filename='Model', infomap_path=infomap_path)

f = open('Model.map', 'r')
line = ''
while '*Nodes' not in line:
    line = f.readline()

line = f.readline()
map_equation = []
map_equation_areas = []
while "*Links" not in line:
    map_equation.append(int(line.split(':')[0]))
    map_equation_areas.append(line.split('"')[1])
    line = f.readline()

# sort map_equation lists
index = []
for i in range(32):
    index.append(map_equation_areas.index(area_list[i]))

map_equation = np.array(map_equation)


"""
Plot graph
"""
plot_clustered_graph(g,
                     g_abs,
                     map_equation[index],
                     'Fig7_community_structure.eps',
                     center_of_masses,
                     colors)


"""
Test significance of clustering using 1000
null models.
"""
null_model = copy.copy(conn_matrix)
mod_list = []

# Shuffling of the connectivity
# In the connectivity matrix, rows == targets, columns == sources
# For each column, we shuffle the rows and therefore conserve the
# total outdegree of each area.

for i in range(1000):
    for j in range(32):
        ind = np.extract(np.arange(32) != j, np.arange(32))
        null_model[:, j][ind] = null_model[:, j][ind][np.random.shuffle(ind)]
    modules, modules_areas, index = apply_map_equation(null_model, area_list,
                                                       filename='null',
                                                       infomap_path=infomap_path)
    g_null = create_graph(null_model, area_list)
    mod_list.append(modularity(g_null, modules[index]))

print("Default structure, map_equation, Q = {}".format(modularity(g,
                                                                  map_equation[index])))
