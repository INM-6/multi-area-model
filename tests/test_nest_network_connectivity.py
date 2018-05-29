from multiarea_model import MultiAreaModel
import nest
import numpy as np


def test_nest_network_connectivity():
    """
    Test if nest creates the correct number of neurons
    synapses using the Simulation class for a downscaled
    network.
    """
    network_params = {'N_scaling': 0.001,
                      'K_scaling': 0.001,
                      'fullscale_rates': 'fullscale_rates.json'}
    M = MultiAreaModel(network_params, simulation=True)
    M.simulation.simulate()

    """
    Test if the correct number of neurons has been created.
    """
    print("Testing neuron numbers")
    for area_name in M.area_list:
        area = M.simulation.areas[M.simulation.areas.index(area_name)]
        for pop in M.structure[area.name]:
            created_nodes = area.gids[pop][1] - area.gids[pop][0] + 1
            assert(created_nodes == int(M.N[area.name][pop]))

    """
    Test if the correct number of synapses has been created.
    """
    print("Testing synapse numbers")
    for target_area_name in M.area_list:
        target_area = M.simulation.areas[M.simulation.areas.index(target_area_name)]
        for source_area_name in M.area_list:
            source_area = M.simulation.areas[M.simulation.areas.index(source_area_name)]
            for target_pop in M.structure[target_area.name]:
                target_gids = list(range(target_area.gids[target_pop][0],
                                         target_area.gids[target_pop][1] + 1))
                for source_pop in M.structure[source_area.name]:
                    source_gids = list(range(source_area.gids[source_pop][0],
                                             source_area.gids[source_pop][1] + 1))
                    created_syn = nest.GetConnections(source=source_gids,
                                                      target=target_gids)
                    syn = M.synapses[target_area.name][target_pop][source_area.name][source_pop]
                    assert(len(created_syn) == int(syn))

    """
    Test if the correct external input has been created.
    """
    print("Testing external input")
    for area in M.simulation.areas:
        poisson_rates = nest.GetStatus(area.poisson_generators, 'rate')

        K_ext = []
        for pop in M.structure[area.name]:
            K_ext.append(M.K[area.name][pop]['external']['external'])
        target_rates = np.array(K_ext) * M.params['input_params']['rate_ext']
        assert(np.allclose(poisson_rates, target_rates))
