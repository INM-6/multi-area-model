import os
import pylab as pl
import numpy as np
from plotcolors import myred, myblue
from matplotlib.colors import ListedColormap
from multiarea_model.theory import Theory
from multiarea_model.theory_helpers import nu0_fb
from multiarea_model.default_params import single_neuron_dict
from multiarea_model.multiarea_helpers import convert_syn_weight


class network:
    def __init__(self, params, theory_spec):
        self.label = '1D'
        self.params = {'input_params': params['input_params'],
                       'neuron_params': {'single_neuron_dict': single_neuron_dict},
                       'connection_params': {'replace_cc': None,
                                             'replace_cc_input_source': None}
                       }

        self.add_DC_drive = np.zeros(1)
        self.structure = {'A': {'E'}}
        self.structure_vec = ['A-E']
        self.area_list = ['A']
        self.K_matrix = np.array([[params['K'], params['K']]])
        self.W_matrix = np.array([[params['W'], params['W']]])
        self.J_matrix = convert_syn_weight(self.W_matrix,
                                           self.params['neuron_params']['single_neuron_dict'])

        self.theory = Theory(self, theory_spec)
        

    def Phi(self, rate):
        mu, sigma = self.theory.mu_sigma(rate)
#        print(mu, sigma)
        NP = self.params['neuron_params']['single_neuron_dict']
        return nu0_fb(mu, sigma,
                      1.e-3*NP['tau_m'],
                      1.e-3*NP['tau_syn_ex'],
                      1.e-3*NP['t_ref'],
                      NP['V_th'] - NP['E_L'],
                      NP['V_reset'] - NP['E_L'])
        

"""
space showing bifurcation
"""

rate_exts_array = np.arange(150., 170.1, 1.)

network_params = {'K': 105.,
                  'W': 40.}
theory_params = {'T': 20.,
                 'dt': 0.01}
                  

# for i, rate_ext in enumerate(rate_exts_array):
#     input_params = {'rate_ext': rate_ext}
#     network_params.update({'input_params': input_params})

#     net = network(network_params, theory_params)
#     r = net.theory.integrate_siegert()[1][:, -1]
#     print(r)

# x = np.arange(0, 30., 0.02)
# for i, rate_ext in enumerate([150., 160., 170.]):
#     input_params = {'rate_ext': rate_ext}
#     network_params.update({'input_params': input_params})
#     net = network(network_params, theory_params)
#     y = np.fromiter([net.Phi(xi) for xi in x], dtype=np.float)
#     pl.plot(x, y)
    
# pl.savefig('Fig2_EE_example_1D_data.eps')
    

fig = pl.figure()    
x = np.arange(0, 30., 0.02)
# x = [30.]
K = [26.25, 52.5, 105., 210., 420.]
W = [160., 80., 40., 20., 10.]
rate_ext = 150.
for k, w in zip(K, W):
    input_params = {'rate_ext': rate_ext}
    network_params.update({'input_params': input_params,
                           'K': k,
                           'W': w})
    net = network(network_params, theory_params)
    y = np.fromiter([net.Phi(xi) for xi in x], dtype=np.float)
    pl.plot(x, y)
    
# for i, dic in enumerate(mfp.par_list(PS)):
#     print(dic)
#     para_dic, label = mfp.hashtag(dic)
#     mf = meanfield_multi(para_dic, label)
#     inits = np.arange(0, 50, 10)
#     Fps = []
#     for init in inits:
#         solution = mf.fsolve([init, init])
#         if solution['eps'] == 'The solution converged.':
#             Fps.append(mf.fsolve([init, init])['rates'][0])
#     Fps = np.unique(np.round(Fps, decimals=2))
#     Fps_array.append(Fps)

# h5.add_to_h5('mf_data.h5', {'EE_example': {
#              'Fps_array': Fps_array}}, 'a', overwrite_dataset=True)


# rate_exts_array = np.arange(150., 170.1, 10.)
# rate_exts = np.array([[x, x] for x in rate_exts_array])
# PS = para.ParameterSpace({  # 'g': para.ParameterRange(np.arange(-3.5,-3.0,0.1)),
#     'g': 1.,
#     'rate': para.ParameterRange(rate_exts),
#     'model': 'Brunel',
#     'gamma': 1.,
#     'rates_init_int': np.array([80, 80]),
#                          'W': 0.01,
#                          'K': 210.,
#                          })
# print(PS)
# cmap = pl.get_cmap('Greys_r')


# ################## rate instability ###############

# x = np.arange(0, 150, 1.0)
# x_long = np.arange(0, 10000, 10.0)
# NUM_COLORS = len(mfp.par_list(PS))


# for i, dic in enumerate(mfp.par_list(PS)):
#     para_dic, label = mfp.hashtag(dic)
#     mf = meanfield_multi(para_dic, label)
#     dic_refrac = copy.deepcopy(dic)
#     dic_refrac.update({'tau_refrac': 0})
#     para_dic_refrac, label_refrac = mfp.hashtag(dic_refrac)
#     mf_refrac = meanfield_multi(para_dic_refrac, label_refrac)
#     t = [mf.Phi(np.array([xval, xval]), return_leak=False) for xval in x]
#     t_noisefree = [mf.Phi_noisefree(np.array([xval, xval])) for xval in x]
#     t_refrac = [mf_refrac.Phi(np.array([xval, xval]),
#                               return_leak=False) for xval in x]

#     t_long = [mf.Phi(np.array([xval, xval]), return_leak=False)
#               for xval in x_long]
#     h5.add_to_h5('mf_data.h5', {'EE_example': {label: {'t': t, 't_noisefree': t_noisefree,
#                                                        't_long': t_long, 't_refrac': t_refrac}}}, 'a', overwrite_dataset=True)

# ########################## Stabilization ################

# x = np.arange(0., 50., 0.1)
# rate = 160.
# start = 17.0
# drate = 1.

# dic = {  # 'g': para.ParameterRange(np.arange(-3.5,-3.0,0.1)),
#     'g': 1.,
#     'rate': np.array([rate, rate]),
#     'model': 'Brunel',
#     'gamma': 1.,
#     'W': 0.01,
#     'K': 210.,
# }

# ######### base state ##########

# para_dic, mf_label = mfp.hashtag(dic)
# mf = meanfield_multi(para_dic, mf_label)
# res_la_base = mf.fsolve(np.ones(2) * start)
# res_in_base = mf.fsolve(np.ones(2) * start)
# print('res', res_la_base)
# t = [mf.Phi(np.array([xval, xval])) for xval in x]
# savedic = {'res_la_base': res_la_base, 'res_in_base': res_in_base, 't': t}
# h5.add_to_h5('mf_data.h5', {'EE_example': {
#              mf_label: savedic}}, 'a', overwrite_dataset=True)

# ######## increased rate #######

# dic.update({'rate': np.array([rate + drate, rate + drate])})
# para_dic, mf_label = mfp.hashtag(dic)
# mf_2 = meanfield_multi(para_dic, mf_label)
# res_in_2 = mf_2.fsolve(np.ones(2) * start)
# t_2 = [mf_2.Phi(np.array([xval, xval])) for xval in x]
# savedic = {'res_in_2': res_in_2, 't_2': t_2}
# h5.add_to_h5('mf_data.h5', {'EE_example': {
#              mf_label: savedic}}, 'a', overwrite_dataset=True)

# ####### stabilized #######

# matrix_prime, v, shift, delta = mft.stabilize(
#     mf, mf_2, fixed_point=res_la_base['rates'][0], method='least_squares')
# print('1D K_prime', matrix_prime)
# dic.update({'K_stable': matrix_prime})
# para_dic, mf_label = mfp.hashtag(dic)
# mf_s = meanfield_multi(para_dic, mf_label)
# res_in_s = mf_s.fsolve(np.ones(2) * start)
# t_s = [mf_s.Phi(np.array([xval, xval])) for xval in x]
# savedic = {'res_in_s': res_in_s, 't_s': t_s}
# h5.add_to_h5('mf_data.h5', {'EE_example': {
#              mf_label: savedic}}, 'a', overwrite_dataset=True)
