import numpy as np
from .multiarea_helpers import create_mask, create_vector_mask
from copy import deepcopy
from .theory_helpers import d_nu_d_mu_fb_numeric, d_nu_d_sigma_fb_numeric
import copy

"""
Implementation of the stabilization method of [1].
The notation follows Eqs. (6-13) of [1].

1. Schuecker J, Schmidt M, van Albada SJ, Diesmann M & Helias M (2017)
   Fundamental Activity Constraints Lead to Specific Interpretations of the Connectome.
   PLOS Computational Biology, 13(2).
   [https://doi.org/10.1371/journal.pcbi.1005179](https://doi.org/10.1371/journal.pcbi.1005179)
"""


def stabilize(theo, theo_prime, fixed_point, a='fac_nu_ext_5E_6E', b='indegree'):
    """
    Implementation of the stabilization algorithm.

    Parameters
    ----------
    theo : Instance of Theory class
        Unperturbed network.
    theo_prime : Instance of Theory class
        Network perturbed by a change in the a parameter
    fixed_point : numpy.ndarray
        Unstable fixed point that we want to preserve.
    a : str
        The first parameter to be changed. Defaults to
        'fac_nu_ext_5E_6E' which is the relative change of the
        external indegree onto populations 5E and 6E.
    b : str
        The second parameter to be changed in order to preserve the
        location of the separatrix. Defaults to the indegrees.
    """
    if b != 'indegree':
        raise NotImplementedError("Stabilizing using b = {} is not implemented.".format(b))

    """
    First calculate the change of the fixed point that, to first
    order, is described by Eq. 6 of [1], using Eq. 8.
    """
    S_vector, S, T_vector, T, M = S_T(theo, fixed_point)
    delta_bar_nu_star = fixed_point_shift(a, theo, theo_prime, fixed_point)
    delta_nu_star = np.dot(np.linalg.inv(np.identity(M.shape[0]) - M), delta_bar_nu_star)

    """
    Next, determine the change of the parameter b that is
    necessary to revert the change (Eq. 9).

    Calculate eigen decomposition of the effective connectivity
    matrix M
    """
    lambda_ev, u, v = eigen_decomp_M(M)

    a_hat = np.dot(v, delta_bar_nu_star)
    v_hat = np.dot(v, fixed_point)
    epsilon = - 1. * a_hat / v_hat

    # Calculate the single terms of the sum in Eq. (13)
    eta_tilde = []
    d = np.dot(delta_nu_star, delta_nu_star)
    for l in range(epsilon.size):
        eta_tilde.append(-1. * a_hat[l] / (1 - lambda_ev[l]) * np.dot(u[:, l], delta_nu_star) / d)

    """
    Calculate perturbation of beta (Eq. 11)
    Only take the most critical eigendirection into account.
    """
    eigen_proj = np.outer(u[:, 0], v[0])
    denom = (S * theo.network.J_matrix[:, :-1] +
             T * theo.network.J_matrix[:, :-1]**2) * theo.NP['tau_m'] * 1.e-3
    delta_K = epsilon[0] * eigen_proj / denom

    """
    Apply three constraints:
    1. No inhibitory cortico-cortical connections
    2. No cortico-cortical connections from population 4E
    3. Indegree have to be > 0 -> Negative entries are set to zero.
    """
    index = np.zeros_like(delta_K, dtype=np.bool)
    for area in theo.network.area_list:
        for area2 in theo.network.area_list:
            if area2 != area:
                mask = create_mask(theo.network.structure,
                                   target_areas=[area],
                                   source_areas=[area2],
                                   source_pops=['23I', '4E', '4I', '5I', '6I'])
                index = np.logical_or(index, mask[:, :-1])
    delta_K[index] = 0.
    K_prime = copy.copy(theo.network.K_matrix)
    K_prime[:, :-1] += np.real(delta_K)
    K_prime[np.where(K_prime < 0.0)] = 0.0
    return K_prime


def S_T(theo, fixed_point):
    mu, sigma = theo.mu_sigma(fixed_point)
    S_vector, T_vector = theo.d_nu(mu, sigma)
    S = np.array([S_vector[i] * np.ones(theo.network.K_matrix.shape[0])
                  for i in range(theo.network.K_matrix.shape[0])])
    T = np.array([T_vector[i] * np.ones(theo.network.K_matrix.shape[0])
                  for i in range(theo.network.K_matrix.shape[0])])
    W = theo.network.K_matrix[:, :-1] * theo.network.J_matrix[:, :-1]
    W2 = theo.network.K_matrix[:, :-1] * theo.network.J_matrix[:, :-1]**2
    M = (S * W * theo.NP['tau_m'] * 1.e-3 +
         T * W2 * theo.NP['tau_m'] * 1.e-3)
    return S_vector, S, T_vector, T, M


def fixed_point_shift(a, theo, theo_prime, fixed_point):
    S_vector, S, T_vector, T, SJ_TJ2 = S_T(theo, fixed_point)
    if a in ['fac_nu_ext_5E_6E']:
        W_ext = deepcopy(theo.network.J_matrix[:, -1])

        K_ext = deepcopy(theo.network.K_matrix[:, -1])
        K_ext_prime = theo_prime.network.K_matrix[:, -1]
        delta_Kext = K_ext_prime - K_ext

        rate_ext = theo.network.params['input_params']['rate_ext']
        v_mu = theo.NP['tau_m'] * 1.e-3 * S_vector * delta_Kext * W_ext * rate_ext
        v_sigma = theo.NP['tau_m'] * 1.e-3 * T_vector * delta_Kext * W_ext**2 * rate_ext
        v = v_mu + v_sigma
    else:
        raise NotImplementedError('a = {} not implemented.'.format(a))
    return v


def eigen_decomp_M(M):
    eig = np.linalg.eig(M)
    evec_left = np.linalg.inv(eig[1])
    evec_right = eig[1]
    evals = eig[0]

    index = np.argsort(np.real(evals))[::-1]
    evals_sorted = evals[index]
    evec_right_sorted = evec_right[:, index]
    evec_left_sorted = evec_left[index]

    return evals_sorted, evec_right_sorted, evec_left_sorted
