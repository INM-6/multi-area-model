from multiarea_model import MultiAreaModel


def test_network_initialization():
    """
    Tests two different ways to initilize a network:
    - From a dictionary of custom parameters
    - From a label string
    Tests whether the two instances yield
    identical networks.
    """
    conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
                   'g': -11.,
                   'K_stable': '../K_stable.npy',
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.41666667,
                   'av_indegree_V1': 3950.}
    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'connection_params': conn_params}

    M = MultiAreaModel(network_params)
    M2 = MultiAreaModel(M.label)
    assert(M == M2)
