import json
import numpy as np
import os
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import vector_to_dict
from multiarea_model.multiarea_helpers import create_mask
from multiarea_model.default_params import complete_area_list, population_list
from multiarea_model.analysis_helpers import _save_dict_to_npy

"""
Test replacing cortico-cortical connections.
"""


def test_het_poisson_stat_mf():
    network_params = {}
    theory_params = {}
    M0 = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
    p, r0 = M0.theory.integrate_siegert()

    rates = vector_to_dict(r0[:, -1], M0.area_list, M0.structure)
    with open('mf_rates.json', 'w') as f:
        json.dump(rates, f)

    network_params = {'connection_params': {'replace_cc': 'het_poisson_stat',
                                            'replace_cc_input_source': 'mf_rates.json'}}
    theory_params = {}
    M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
    p, r = M.theory.integrate_siegert()

    assert(np.allclose(r0[:, -1], r[:, -1]))


def test_hom_poisson_stat_mf():
    network_params = {'connection_params': {'replace_cc': 'hom_poisson_stat'}}
    theory_params = {}
    M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
    p, r = M.theory.integrate_siegert()

    mu, sigma = M.theory.replace_cc_input()
    # Test for V1
    mask = create_mask(M.structure,
                       target_areas=['V1'],
                       cortico_cortical=True,
                       external=False)
    x = np.sum((M.J_matrix[mask].reshape((8, -1)) * M.K_matrix[mask].reshape((8, -1)) *
                M.params['input_params']['rate_ext'] *
                M.params['neuron_params']['single_neuron_dict']['tau_m'] * 1e-3), axis=1)
    assert(np.allclose(x, mu[:8]))


def test_het_poisson_stat_sim():
    base_dir = os.getcwd()
    fn = os.path.join(base_dir, 'fullscale_rates.json')
    network_params = {'connection_params': {'replace_cc': 'het_poisson_stat',
                                            'replace_cc_input_source': fn},
                      'N_scaling': 0.001,
                      'K_scaling': 0.0001,
                      'fullscale_rates': 'fullscale_rates.json'}
    sim_params = {'t_sim': 0.1}
    M = MultiAreaModel(network_params, simulation=True, sim_spec=sim_params)
    M.simulation.simulate()


def test_hom_poisson_stat_sim():
    network_params = {'connection_params': {'replace_cc': 'hom_poisson_stat'},
                      'N_scaling': 0.001,
                      'K_scaling': 0.0001,
                      'fullscale_rates': 'fullscale_rates.json'}
    sim_params = {'t_sim': 0.1}
    M = MultiAreaModel(network_params, simulation=True, sim_spec=sim_params)
    M.simulation.simulate()


def test_het_current_non_stat_sim():
    curr = np.ones(10) * 10.
    het_current = {area: {pop: curr for pop in population_list} for area in complete_area_list}
    _save_dict_to_npy('het_current', het_current)

    base_dir = os.getcwd()
    fs = os.path.join(base_dir, 'het_current')
    network_params = {'connection_params': {'replace_cc': 'het_current_nonstat',
                                            'replace_cc_input_source': fs},
                      'N_scaling': 0.001,
                      'K_scaling': 0.0001,
                      'fullscale_rates': 'fullscale_rates.json'}
    sim_params = {'t_sim': 10.}
    M = MultiAreaModel(network_params, simulation=True, sim_spec=sim_params)
    M.simulation.simulate()
