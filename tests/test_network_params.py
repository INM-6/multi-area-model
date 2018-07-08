from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_mask
import numpy as np
from multiarea_model.default_params import av_indegree_Cragg, av_indegree_OKusky


def create_default_network():
    """
    Return an instance of the default network.
    """
    network_params = {}
    M0 = MultiAreaModel(network_params)
    return M0


def test_average_indegree():
    """
    Test different average indegrees.
    """
    for av_indegree in [av_indegree_Cragg,
                        np.mean([av_indegree_Cragg, av_indegree_OKusky]),
                        av_indegree_OKusky]:
        conn_params = {'av_indegree_V1': av_indegree}
        network_params = {'connection_params': conn_params}
        M = MultiAreaModel(network_params)

        area = 'V1'
        mask = create_mask(M.structure, target_areas=[area])
        x = np.sum(M.syn_matrix[mask].reshape((8, 255)))
        K_average = x / M.N[area]['total']
        print(K_average)
        assert(np.allclose(K_average, conn_params['av_indegree_V1']))


def test_external_indegrees():
    """
    Test settings for external indegrees.
    """
    M0 = create_default_network()
    K0 = M0.K_matrix
    conn_params = {'fac_nu_ext_5E': 2.,
                   'fac_nu_ext_6E': 2.}
    network_params = {'connection_params': conn_params}
    M = MultiAreaModel(network_params)

    mask = create_mask(M.structure, target_pops=['5E'], source_pops=[])
    assert(np.allclose(M.K_matrix[mask], conn_params['fac_nu_ext_5E'] * K0[mask]))
    mask = create_mask(M.structure, target_pops=['6E'], source_pops=[])
    assert(np.allclose(M.K_matrix[mask], conn_params['fac_nu_ext_6E'] * K0[mask]))

    conn_params = {'fac_nu_ext_TH': 2.}
    network_params.update({'connection_params': conn_params})
    M = MultiAreaModel(network_params)

    mask = create_mask(M.structure,
                       target_areas=['TH'],
                       target_pops=['23E', '5E'],
                       source_pops=[])

    assert(np.allclose(M.K_matrix[mask], conn_params['fac_nu_ext_TH'] * K0[mask]))


def test_syn_weights():
    """
    Test different options for synaptic weights.
    """
    M0 = create_default_network()
    W0 = M0.W_matrix

    conn_params = {'PSP_e': 0.3,
                   'PSP_e_23_4': 0.6}
    network_params = {'connection_params': conn_params}
    M = MultiAreaModel(network_params)
    mask = create_mask(M.structure,
                       source_pops=['23E', '4E', '5E', '6E'],
                       external=False)
    assert(np.allclose(M.W_matrix[mask], conn_params['PSP_e'] / 0.15 * W0[mask]))

    conn_params = {'cc_weights_factor': 2.,
                   'cc_weights_I_factor': 2.}
    network_params.update({'connection_params': conn_params})
    M = MultiAreaModel(network_params)
    mask = create_mask(M.structure,
                       source_pops=['23E', '5E', '6E'],
                       target_pops=['23E', '4E', '5E', '6E'],
                       cortico_cortical=True)
    assert(np.allclose(M.W_matrix[mask], conn_params['cc_weights_factor'] * W0[mask]))

    mask = create_mask(M.structure,
                       source_pops=['23E', '5E', '6E'],
                       target_pops=['23I', '4I', '5I', '6I'],
                       cortico_cortical=True)
    assert(np.allclose(M.W_matrix[mask], conn_params['cc_weights_factor'] *
                       conn_params['cc_weights_I_factor'] * W0[mask]))
