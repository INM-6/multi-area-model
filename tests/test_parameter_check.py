import pytest
from multiarea_model import Model

"""
Test if the check for unused keys in
custom parameters works.
"""


def test_network_params():
    net_params = {'x': 3}
    with pytest.raises(KeyError):
        Model(net_params)


def test_conn_params():
    net_params = {'connection_params': {'x': 3}}
    with pytest.raises(KeyError):
        Model(net_params)


def test_sim_params():
    sim_params = {'x': 3}
    with pytest.raises(KeyError):
        Model({}, simulation=True, sim_spec=sim_params)


def test_theory_params():
    theory_params = {'x': 3}
    with pytest.raises(KeyError):
        Model({}, theory=True, theory_spec=theory_params)
