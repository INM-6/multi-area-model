from multiarea_model import MultiAreaModel


def test_meanfield():
    """
    Test meanfield calculation of the
    stationary network state.
    """

    network_params = {}
    theory_params = {}
    M0 = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
    p, r0 = M0.theory.integrate_siegert()
