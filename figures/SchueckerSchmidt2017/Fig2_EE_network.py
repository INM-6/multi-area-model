import numpy as np

from scipy import optimize
from multiarea_model.theory import Theory
from multiarea_model.theory_helpers import nu0_fb
from multiarea_model.default_params import single_neuron_dict, nested_update
from multiarea_model.multiarea_helpers import convert_syn_weight
from copy import copy

"""
Network class for the 1D case:
1 excitatory population with recurrent connectivity and external
stimulation.
"""


class network1D:
    def __init__(self, params):
        self.label = '1D'
        self.params = {'input_params': params['input_params'],
                       'neuron_params': {'single_neuron_dict': copy(single_neuron_dict)},
                       'connection_params': {'replace_cc': None,
                                             'replace_cc_input_source': None}
                       }
        nested_update(self.params, params)
        self.add_DC_drive = np.zeros(1)
        self.structure = {'A': {'E'}}
        self.structure_vec = ['A-E']
        self.area_list = ['A']
        if 'K_stable' in params.keys():
            self.K_matrix = np.array([[params['K_stable'], params['K']]])
        else:
            self.K_matrix = np.array([[params['K'], params['K']]])

        self.W_matrix = np.array([[params['W'], params['W']]])
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])
        self.theory = Theory(self, {})

    def Phi(self, rate):
        mu, sigma = self.theory.mu_sigma(rate)
        NP = self.params['neuron_params']['single_neuron_dict']
        return list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                 1.e-3 * NP['tau_m'],
                                                 1.e-3 * NP['tau_syn_ex'],
                                                 1.e-3 * NP['t_ref'],
                                                 NP['V_th'] - NP['E_L'],
                                                 NP['V_reset'] - NP['E_L']),
                        mu, sigma))

    def Phi_noisefree(self, rate):
        mu, sigma = self.theory.mu_sigma(rate)
        NP = self.params['neuron_params']['single_neuron_dict']
        th_shift = NP['V_th'] - NP['E_L']
        if mu > th_shift:
            T = 1e-3 * NP['tau_m'] * \
                np.log(mu[0] / (mu[0] - th_shift))
            return (1 / T)
        else:
            return 0.

    def fsolve(self, rates_init):
        def f(rate):
            return self.Phi(rate) - rate
        result = optimize.fsolve(f, rates_init, full_output=1)
        mu, sigma = self.theory.mu_sigma(result[0])
        result_dic = {'rates': np.array([result[0]]), 'mus': np.array(
            [mu]), 'sigmas': np.array([sigma]), 'eps': result[-1], 'time': np.array([0])}
        return result_dic


"""
Network class for the 2D case:
2 excitatory populations with recurrent connectivity and external
stimulation.
"""


class network2D:
    def __init__(self, params):
        self.label = '2D'
        self.params = {'input_params': params['input_params'],
                       'neuron_params': {'single_neuron_dict': copy(single_neuron_dict)},
                       'connection_params': {'replace_cc': None,
                                             'replace_cc_input_source': None}
                       }
        nested_update(self.params, params)
        self.add_DC_drive = np.zeros(1)
        self.structure = {'A': {'E1', 'E2'}}
        self.structure_vec = ['A-E1', 'A-E2']
        self.area_list = ['A']
        if 'K_stable' in params.keys():
            self.K_matrix = np.array(
                [[params['K_stable'] / 2., params['K_stable'] / 2., params['K']]])
        else:
            self.K_matrix = np.array(
                [[params['K'] / 2., params['K'] / 2., params['K']]])

        self.W_matrix = np.array([[params['W'], params['W'], params['W']]])
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])
        self.theory = Theory(self, {})

    def Phi(self, rate):
        mu, sigma = self.theory.mu_sigma(rate)
        NP = self.params['neuron_params']['single_neuron_dict']
        return list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                 1.e-3 * NP['tau_m'],
                                                 1.e-3 * NP['tau_syn_ex'],
                                                 1.e-3 * NP['t_ref'],
                                                 NP['V_th'] - NP['E_L'],
                                                 NP['V_reset'] - NP['E_L']),
                        mu, sigma))

    def fsolve(self, rates_init):
        def f(rate):
            return self.Phi(rate) - rate
        result = optimize.fsolve(f, rates_init, full_output=1)
        mu, sigma = self.theory.mu_sigma(result[0])
        result_dic = {'rates': np.array([result[0]]), 'mus': np.array(
            [mu]), 'sigmas': np.array([sigma]), 'eps': result[-1], 'time': np.array([0])}
        return result_dic

    def vector_field(self, x_vec, y_vec):
        NP = self.params['neuron_params']['single_neuron_dict']
        vector_matrix_x = np.zeros((len(y_vec), len(x_vec)))
        vector_matrix_y = np.zeros((len(y_vec), len(x_vec)))
        for i, x in enumerate(y_vec):
            for j, y in enumerate(x_vec):
                mu, sigma = self.theory.mu_sigma([x, y])
                new_rates = np.array(
                    list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                      1.e-3 * NP['tau_m'],
                                                      1.e-3 * NP['tau_syn_ex'],
                                                      1.e-3 * NP['t_ref'],
                                                      NP['V_th'] - NP['E_L'],
                                                      NP['V_reset'] - NP['E_L']),
                             mu, sigma)))
                vector_matrix_x[i, j] = (new_rates[1] - y)
                vector_matrix_y[i, j] = (new_rates[0] - x)
        x, y = np.meshgrid(x_vec, y_vec)
        return x, y, vector_matrix_x, vector_matrix_y

    def nullclines_x0(self, x0_vec):
        NP = self.params['neuron_params']['single_neuron_dict']

        def nullcline(x0, x1):
            rates = np.zeros(2)
            rates[0] = x0
            rates[1] = x1
            mu, sigma = self.theory.mu_sigma(rates)
            new_rates = np.array(
                list(map(lambda mu, sigma: nu0_fb(mu, sigma,
                                                  1.e-3 * NP['tau_m'],
                                                  1.e-3 * NP['tau_syn_ex'],
                                                  1.e-3 * NP['t_ref'],
                                                  NP['V_th'] - NP['E_L'],
                                                  NP['V_reset'] - NP['E_L']), mu, sigma)))[0]
            return new_rates - x0

        nullcline_x0 = []
        for x0 in x0_vec:
            result = optimize.fsolve(
                lambda x: nullcline(x0, x), 0, full_output=1)
            nullcline_x0.append(result[0][0])
        return nullcline_x0
