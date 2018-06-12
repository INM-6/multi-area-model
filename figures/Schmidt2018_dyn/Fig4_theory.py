import os
import sys
import numpy as np
from multiarea_model import MultiAreaModel

K_stable_path = '../SchueckerSchmidt2017/K_prime_original.npy'

cc_weights_factor = float(sys.argv[1])

if cc_weights_factor == 1.:
    cc_weights_I_factor = 1.
else:
    cc_weights_I_factor = 2.
    
conn_params = {'g': -11.,
               'cc_weights_factor': cc_weights_factor,
               'cc_weights_I_factor': cc_weights_I_factor,
               'K_stable': K_stable_path,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.125 * 10 / 3. - 7 / 3.,
               'fac_nu_ext_TH': 1.2}
input_params = {'rate_ext': 10.}
network_params = {'connection_params': conn_params,
                  'input_params': input_params}

initial_rates = np.zeros(254)
theory_params = {'T': 30.,
                 'dt': 0.01,
                 'rec_interval': 30.,
                 'initial_rates': 'random_uniform',
                 'initial_rates_iter': 1000}

M = MultiAreaModel(network_params, theory=True,
                   theory_spec=theory_params)
p, r_base = M.theory.integrate_siegert()
np.save(os.path.join('Fig4_theory_data',
                     'results_{}.npy'.format(cc_weights_factor)),
        r_base)
