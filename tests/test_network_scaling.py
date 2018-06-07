from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import vector_to_dict
import numpy as np
import json


def test_network_scaling():
    """
    Test the downscaling option of the network.

    - Test whether indegrees and neuron number are correctly scaled down.
    - Test whether the resulting mean and variance of the input currents
      as well as the resulting rates are identical, based on mean-field theory.
    """

    network_params = {}
    M0 = MultiAreaModel(network_params, theory=True)
    K0 = M0.K_matrix
    W0 = M0.W_matrix
    N0 = M0.N_vec
    syn0 = M0.syn_matrix
    p, r0 = M0.theory.integrate_siegert()

    d = vector_to_dict(r0[:, -1],
                       M0.area_list,
                       M0.structure)

    with open('mf_rates.json', 'w') as f:
        json.dump(d, f)

    network_params = {'N_scaling': .1,
                      'K_scaling': .1,
                      'fullscale_rates': 'mf_rates.json'}
    theory_params = {'initial_rates': r0[:, -1],
                     'T': 50.}
    M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)

    K = M.K_matrix
    W = M.W_matrix
    N = M.N_vec
    syn = M.syn_matrix
    p, r = M.theory.integrate_siegert()

    assert(np.allclose(K, network_params['K_scaling'] * K0))
    assert(np.allclose(N, network_params['N_scaling'] * N0))
    assert(np.allclose(syn, network_params['K_scaling'] * network_params['N_scaling'] * syn0))
    assert(np.allclose(W, W0 / np.sqrt(network_params['K_scaling'])))

    r0_extend = np.append(r0[:, -1], M0.params['input_params']['rate_ext'])
    tau_m = M.params['neuron_params']['single_neuron_dict']['tau_m']
    C_m = M.params['neuron_params']['single_neuron_dict']['C_m']

    mu0 = (1e-3 * tau_m * np.dot(M0.K_matrix * M0.J_matrix, r0_extend)
           + tau_m / C_m * M0.add_DC_drive)
    mu = 1e-3 * tau_m * np.dot(M.K_matrix * M.J_matrix, r0_extend) + tau_m / C_m * M.add_DC_drive

    sigma0 = np.sqrt(1e-3 * tau_m * np.dot(M0.K_matrix * M0.J_matrix**2, r0_extend))
    sigma = np.sqrt(1e-3 * tau_m * np.dot(M.K_matrix * M.J_matrix**2, r0_extend))

    assert(np.allclose(mu, mu0))
    assert(np.allclose(sigma, sigma0))
    assert(np.allclose(r[:, -1], r0[:, -1]))
