from multiarea_model import MultiAreaModel
import pytest


def test_meanfield():
    """
    Test stabilization procedure. Since this algorithm is not
    implemented yet, we here test if this properly raises a
    NotImplementedError.
    """

    network_params = {'connection_params': {'K_stable': True}}
    theory_params = {}
    with pytest.raises(NotImplementedError):
        MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
