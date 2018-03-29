import copy
import igraph
import networkx as nx
import numpy as np
import os


def perform_map_equation(graph_matrix, node_list, filename='', infomap_path=None):
    """
    Perform the map equation of
    Rosvall M, Axelsson D, Bergstrom CT (2009) The map equation.
    Eur Phys J Spec Top 178(1):13–23

    Parameters
    ----------
    graph_matrix : numpy.ndarray
        Matrix of edge weights.
    node_list : array-like
        List of vertex names.
        The ordering of vertices has to be consistent in
        the graph_matrix and node_list.
    filename : str
        Output file name without ending.
    infomap_path : str
        Path to installation of infomap. If None, the
        infomap executable has to be in the system path.

    Returns
    -------
    map_equation : numpy.ndarray
        List of cluster indices
    map_equation_areas: numpy.ndarray
        List of areas in the same order as in
        map_equation
    index : numpy.ndarray
        Index to map node_list to map_equation_areas
    """
    base_dir = os.getcwd()
    """
    1. write out graph to file for map equation
    """
    net_fn = '{}.net'.format(filename)
    # nodes
    f = open(net_fn, 'w')
    f.write('*Vertices ' + str(len(node_list)) + '\n')
    for ii, area in enumerate(node_list):
        f.write(str(ii + 1) + ' "' + area + '"\n')

    # Determine number of vertices in the network
    k = np.where(graph_matrix != 0)[0].size
    f.write('*Arcs ' + str(k) + '\n')

    # edges
    for ii in range(len(node_list)):
        for jj in range(len(node_list)):
            if graph_matrix[ii][jj] > 0.:
                f.write(str(jj + 1) + ' ' + str(ii + 1) +
                        ' ' + str(graph_matrix[ii][jj]) + '\n')
    f.close()

    """
    2. Execute map equation algorithm
    """
    if infomap_path:
        os.chdir(infomap_path)
    os.system('./Infomap --directed --clu --map --verbose ' +
              base_dir + '/' + net_fn + ' ' + base_dir)

    os.chdir(base_dir)

    """
    3. Parse results of map equation
    """
    map_fn = '{}.map'.format(filename)
    f = open(map_fn, 'r')
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
    for ii in range(32):
        index.append(map_equation_areas.index(node_list[ii]))

    map_equation = np.array(map_equation)

    return map_equation, map_equation_areas, index


def modularity(g, membership):
    """
    Compute modularity for a given igraph object
    and cluster memberships.

    See Schmidt, M., Bakker, R., Hilgetag, C.C. et al.
    Brain Structure and Function (2017),
    for a derivation.

    Parameters
    ----------
    g : igraph.Graph
        Graph object
    membership : array-like
        List of cluster memberships.

    Returns
    -------
    Q : float
        Modularity value.
    """
    m = np.sum(g.es['weight'])

    Q = 0.
    for ii, area in enumerate(g.vs):
        k_out = np.sum(g.es.select(_source=ii)['weight'])
        for jj, area2 in enumerate(g.vs):
            k_in = np.sum(g.es.select(_target=jj)['weight'])
            if membership[ii] == membership[jj]:
                weight = g.es.select(_source=ii, _target=jj)['weight']
                if len(weight) > 0:
                    Q += weight[0] - k_out * k_in / m
                else:
                    Q += 0. - k_out * k_in / m
    Q /= m
    return Q


def all_pairs_bellman_ford_path(g, weight='distance'):
    """
    Compute the shorted paths between nodes of a graph
    using the Bellman-Ford algorithm.
    See Schmidt, M., Bakker, R., Hilgetag, C.C. et al.
    Brain Structure and Function (2017),
    for details.

    Parameters
    ----------
    g : networkx.graph
        Graph object.
    weight : str
        Edge attributes used for path calculation.
        Defaults to 'weight'.

    Returns
    -------
    paths : dict
        Dictionary of all shortest paths.
    path_lengths: dict
        Dictionary of all shortest path lengths.
    """
    sources = g.nodes()
    predecessors = {}
    path_lengths = {}
    for node in sources:
        res = nx.bellman_ford_predecessor_and_distance(g, node, weight=weight)
        predecessors[node] = res[0]
        path_lengths[node] = res[1]
    paths = {}
    for source in sources:
        paths[source] = {}
        for target in predecessors[source]:
            if target != source:
                path = []
                predec = target
                while predec != source:
                    path.insert(0, predec)
                    predec = predecessors[source][predec][0]
                path.insert(0, predec)
            else:
                path = [source]
            paths[source][target] = path
    return paths, path_lengths


def plot_clustered_graph(g, g_abs, membership, filename, center_of_masses, colors):
    """
    Create a plot of a given graph with nodes arranged in clusters.

    Parameters
    ----------
    g : igraph.Graph
        Graph to be plotted
    membership : list
        List of associated clusters for each area
    filename : str
        Name of file to save the plot.
    center_of_masses : list
        List of 2D cooordinates specifying the center of mass
        of each cluster
    colors : list
        List of colors in hexadecimal format specifying the color
        of each cluster
    """

    # Copy the graphs for further modification necessary for plotting
    gplot = g.copy()
    gplot_abs = g_abs.copy()
    gcopy = g.copy()
    gplot.delete_edges(None)
    gplot_abs.delete_edges(None)
    edges = []
    for edge in g.es():
        if (membership[edge.tuple[0]] != membership[edge.tuple[1]] or
            len(g.es.select(_source=edge.tuple[0],
                            _target=edge.tuple[1])['weight']) == 0):
            edges.append(edge)

    gcopy.delete_edges(edges)

    # Inter-cluster connections are gray
    # Intra-cluster connections are black
    edges_colors = []
    for edge_id, edge in enumerate(g.es()):
        if membership[edge.tuple[0]] != membership[edge.tuple[1]] and edge['weight'] > 0.001:
            gplot.add_edge(edge.tuple[0], edge.tuple[1], weight=edge['weight'])
            gplot_abs.add_edge(edge.tuple[0], edge.tuple[1],
                               weight=g_abs.es()[edge_id]['weight'])
            edges_colors.append("gray")
    for edge_id, edge in enumerate(g.es()):
        if membership[edge.tuple[0]] == membership[edge.tuple[1]] and edge['weight'] > 0.001:
            gplot.add_edge(edge.tuple[0], edge.tuple[1], weight=edge['weight'])
            gplot_abs.add_edge(edge.tuple[0], edge.tuple[1],
                               weight=g_abs.es()[edge_id]['weight'])
            edges_colors.append("black")

    # Inside cluster, distribute areas using a force-directed algorithm
    # by Kamada and Kawai, 1989
    layout_params = {'maxy': list(range(32))}
    layout = gcopy.layout("kk", **layout_params)
    coords = np.array(copy.copy(layout.coords))
    # For better visibility, place clusters at defined positions
    for ii in range(np.max(membership)):
        coo = coords[np.where(membership == ii + 1)]
        com = np.mean(coo, axis=0)
        coo = np.array(coo) - (com - center_of_masses[ii])
        coords[np.where(membership == ii + 1)] = coo
    # Define layout parameters
    gplot.es["color"] = edges_colors
    visual_style = {}
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_shape"] = "circle"
    visual_style["edge_color"] = gplot.es["color"]
    # visual_style["bbox"] = (4000, 2500)
    visual_style["vertex_size"] = 40 * 0.48
    visual_style["layout"] = list(coords)
    visual_style["bbox"] = (493, 380)
    visual_style["margin"] = 22.
    visual_style["vertex_label"] = gplot.vs["name"]
    visual_style["vertex_label_size"] = 7.
    visual_style["edge_arrow_width"] = 2.
    visual_style["vertex_label_color"] = '#ffffff'
    visual_style["edge_arrow_size"] = 0.5

    weights_transformed = np.log(gplot_abs.es['weight']) * 0.48
    visual_style["edge_width"] = weights_transformed

    for vertex in gplot.vs():
        vertex["label"] = vertex.index
    if membership is not None:
        for vertex in gplot.vs():
            vertex["color"] = colors[membership[vertex.index] - 1]
        visual_style["vertex_color"] = gplot.vs["color"]

    # use igraph's plot function to finally plot the graph and save to file
    igraph.plot(gplot, filename, **visual_style)


def create_graph(matrix, area_list):
    """
    Create igraph.Graph instance from a given connectivity matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix
    area_list: list or numpy.ndarray
        List of areas

    Returns
    -------
    g : igraph.Graph
    """
    g = igraph.Graph(directed=True)
    g.add_vertices(area_list)

    for ii in range(32):
        for jj in range(32):
            if matrix[ii][jj] != 0:
                g.add_edge(jj, ii, weight=matrix[ii][jj])
    return g


def create_networkx_graph(matrix, complete_population_list, relative=False):
    """
    Create instance of networkx.DiGraph from connectivity matrix

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix defining the edge weights
    complete_population_list : list
        List of all populations in the model defining
        the graph nodes
    relative : boolean
        Whether to use the given number in matrix
        or relative values for each row of the matrix
        as edge weights. Defaults to False.

    Returns
    -------
    g : networkx.DiGraph
        The resulting graph
    """
    g = nx.DiGraph()
    g.add_nodes_from(complete_population_list)

    for ii, target in enumerate(complete_population_list):
        for jj, source in enumerate(complete_population_list):
            if matrix[ii][jj] != 0:
                if relative:
                    weight = matrix[ii][jj] / np.sum(matrix[ii][:-1])
                else:
                    weight = matrix[ii][jj]
                g.add_edge(source, target, weight=weight,
                           distance=np.log10(1. / weight))
    return g


def create_networkx_area_graph(matrix, area_list, relative=False):
    """
    Create instance of networkx.DiGraph from connectivity matrix

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix defining the edge weights
    area_list : list
        List of all populations in the model defining
        the graph nodes
    relative : boolean
        Whether to use the given number in matrix
        or relative values for each row of the matrix
        as edge weights. Defaults to False.

    Returns
    -------
    g : networkx.DiGraph
        The resulting graph
    """
    G = nx.DiGraph()
    G.add_nodes_from(area_list)

    for ii, target in enumerate(area_list):
        for jj, source in enumerate(area_list):
            if matrix[ii][jj] > 0:
                if relative:
                    weight = matrix[ii][jj] / np.sum(matrix[ii])
                else:
                    weight = matrix[ii][jj]
                G.add_edge(source, target, weight=weight,
                           distance=np.log10(1. / weight))
    return G
