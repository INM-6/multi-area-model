import numpy as np
import pytest

from multiarea_model import MultiAreaModel


def test_stabilization():
    """
    Test stabilization procedure. The stabilized matrix is expected to
    be stored in a file and the parameter in the dictionary specifies
    the corresponding name. We here check if the MultiAreaModel class
    properly throws a TypeError when we try to directly specify the
    matrix.
    """

    # Create random matrix for indegrees
    K_stable = np.random.rand(254, 254)
    np.save('K_stable_test.npy', K_stable)

    # Trying to directly specify the matrix should throw a TypeError.
    network_params = {'connection_params': {'K_stable': K_stable}}
    theory_params = {}
    with pytest.raises(TypeError):
        MultiAreaModel(network_params, theory=True, theory_spec=theory_params)

    # Specifying the file name leads to the correct indegrees being loaded.
    network_params = {'connection_params': {'K_stable': 'K_stable_test.npy'}}
    theory_params = {}
    M = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
    assert(np.all(K_stable == M.K_matrix[:, :-1]))
