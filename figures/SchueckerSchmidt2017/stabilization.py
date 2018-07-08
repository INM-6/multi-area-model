import copy
import pylab as pl
import numpy as np
from multiarea_model import MultiAreaModel
from multiarea_model.multiarea_helpers import create_vector_mask
from multiarea_model.stabilize import stabilize
import utils

"""
Initialization
"""
conn_params = {'g': -16.,
               'av_indegree_V1': 3950.,
               'fac_nu_ext_TH': 1.2}
input_params = {'rate_ext': 8.}

network_params = {'connection_params': conn_params,
                  'input_params': input_params}
theory_params = {'dt': 0.01,
                 'T': 30.}
time = np.arange(0., theory_params['T'], theory_params['dt'])

M_base = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)


c_target = copy.deepcopy(conn_params)
c_target.update({'fac_nu_ext_5E': 1.2,
                'fac_nu_ext_6E': 10/3.*1.2-7/3.})
network_params_target = {'connection_params': c_target,
                         'input_params': input_params}
M_target = MultiAreaModel(network_params_target, theory=True,
                          theory_spec=theory_params)

THREADS = 4
load_list = []

# This list defines which of the detected minima of the velocity
# vector is identified as the unstable fixed point. It has to be
# created manually.
ind_list = [1, 1, 0, 1]

"""
Main loop
"""
data = {}
for iteration in [1, 2, 3, 4, 5]:
    print("Iteration {}".format(iteration))
    if iteration == 1:
        K_stable = None
    else:
        K_stable = 'iteration_{}/K_prime.npy'.format(iteration - 1)
    conn_params = {'g': -16.,
                   'av_indegree_V1': 3950.,
                   'fac_nu_ext_TH': 1.2,
                   'K_stable': K_stable}
    network_params = {'connection_params': conn_params,
                      'input_params': input_params}

    if iteration in load_list:
        data[iteration] = utils.load_iteration(iteration)
    else:
        fac_nu_ext_5E_list = np.append(np.arange(1., 1.2, 0.01), np.array([1.125]))
        
        # Prepare base instance of the network
        M_base = MultiAreaModel(network_params, theory=True, theory_spec=theory_params)
        # Scan parameter space to find a good approximation of the
        # critical parameter value where the model crosses the
        # separatrix for the initial condition of zero rates
        if iteration < 5:
            # For iteration 5, we just analyze the behavior without performing the stabilization
            data[iteration] = utils.compute_iteration(7, fac_nu_ext_5E_list,
                                                      theory_params, M_base, threads=THREADS)
        else:
            data[iteration] = utils.compute_iteration(1, fac_nu_ext_5E_list,
                                                      theory_params, M_base, threads=THREADS)
            
    if iteration != 5:
        # Determine the transition parameter and the minima of the
        # velocity of the trajectories (see Fig. 4 of Schuecker, Schmidt et al.)
        (par_transition, r_low, r_high,
         minima_low, minima_high) = utils.determine_velocity_minima(time, data[iteration])
        # Retrieve unstable fixed point for both trajectories
        unstable_low = r_low[:, minima_low[ind_list[iteration - 1]]]
        unstable_high = r_high[:, minima_high[ind_list[iteration - 1]]]
        # Simple check to guarantee that the unstable fixed point is
        # approximately equal for both trajectories
        assert(np.allclose(unstable_low, unstable_high, rtol=2e-1))

        c = copy.deepcopy(conn_params)
        c.update({'fac_nu_ext_5E': par_transition,
                  'fac_nu_ext_6E': 10/3.*par_transition-7/3.})
        network_params = {'connection_params': c,
                          'input_params': input_params}
        M = MultiAreaModel(network_params, theory=True,
                           theory_spec=theory_params)

        K_prime = stabilize(M.theory,
                            M_target.theory,
                            unstable_low,
                            a='fac_nu_ext_5E_6E', b='indegree')
        data[iteration]['K_prime'] = K_prime[:, :-1]
    utils.save_iteration(iteration, data[iteration])


fig = pl.figure()

mask = create_vector_mask(M.structure, pops=['5E', '6E'])
for iteration in [1, 2, 3, 4]:
    pl.plot(data[iteration]['parameters'],
            np.mean(data[iteration]['results'][:, mask, -1], axis=1), '.-')
pl.yscale('Log')
pl.show()
