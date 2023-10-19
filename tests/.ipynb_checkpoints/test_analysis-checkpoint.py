import os
import sys
from multiarea_model import MultiAreaModel
from io import StringIO

"""
Test analysis class:
Run a small simulation, load data, compute all measures
available in Analysis, save data and try load again.
"""


def test_analysis():
    base_dir = os.getcwd()
    fn = os.path.join(base_dir, 'fullscale_rates.json')
    network_params = {'connection_params': {'replace_non_simulated_areas': 'het_poisson_stat',
                                            'replace_cc_input_source': fn},
                      'N_scaling': 0.001,
                      'K_scaling': 0.0001,
                      'fullscale_rates': 'fullscale_rates.json'}
    sim_params = {'t_sim': 500.,
                  'areas_simulated': ['V1', 'V2']}
    M = MultiAreaModel(network_params, simulation=True, sim_spec=sim_params)
    M.simulation.simulate()
    M = MultiAreaModel(network_params, simulation=True, sim_spec=sim_params, analysis=True)

    M.analysis.create_pop_rates(t_min=100.)
    M.analysis.create_pop_rate_dists(t_min=100.)
    M.analysis.create_synchrony(t_min=100.)
    M.analysis.create_rate_time_series(t_min=100.)
    M.analysis.create_synaptic_input(t_min=100.)
    M.analysis.create_pop_cv_isi(t_min=100.)
    M.analysis.create_pop_LvR(t_min=100.)

    M.analysis.save()
    out = StringIO()
    sys.stdout = out
    M.analysis.create_pop_rates(t_min=100.)
    M.analysis.create_pop_rate_dists(t_min=100.)
    M.analysis.create_synchrony(t_min=100.)
    M.analysis.create_rate_time_series(t_min=100.)
    M.analysis.create_synaptic_input(t_min=100.)
    M.analysis.create_pop_cv_isi(t_min=100.)
    M.analysis.create_pop_LvR(t_min=100.)
    sys.stdout = sys.__stdout__
    val = out.getvalue()
    assert(val.count("Loading data from") == 9)
