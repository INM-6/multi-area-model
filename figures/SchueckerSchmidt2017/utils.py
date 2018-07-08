import copy
import multiprocessing as mp
import numpy as np
import os
import pylab as pl
from functools import partial

from multiarea_model.multiarea_helpers import create_mask
from scipy.signal import find_peaks_cwt


def integrate(f5, M_base):
    mask5 = create_mask(M_base.structure, target_pops=['5E'],
                        source_areas=[], external=True)
    mask6 = create_mask(M_base.structure, target_pops=['6E'],
                        source_areas=[], external=True)

    f6 = 10/3.*f5-7/3.
    conn_params = copy.deepcopy(M_base.params['connection_params'])
    conn_params.update({'fac_nu_ext_5E': f5,
                        'fac_nu_ext_6E': f6})
    M = copy.deepcopy(M_base)
    M.K_matrix[mask5] *= f5
    M.K_matrix[mask6] *= f6
    p, r = M.theory.integrate_siegert()
    return r


def iteration_results(fac_nu_ext_5E_list, M_base, threads=None):
    if threads is None:
        results = np.array([integrate(f5, M_base) for f5 in fac_nu_ext_5E_list])
    else:
        integrate_part = partial(integrate,
                                 M_base=M_base)
        pool = mp.Pool(processes=threads)
        results = np.array(pool.map(integrate_part, fac_nu_ext_5E_list))
        pool.close()
        pool.join()
    return results


def velocity_peaks(time, result, threshold=0.05):
    d_nu = np.abs(np.diff(np.mean(result, axis=0)) / np.diff(time)[0])
    ind = np.where(d_nu < threshold)
    minima = find_peaks_cwt(-1. * np.log(d_nu[ind]), np.array([0.1]))
    if len(minima) > 0:
        t_min = time[ind][minima]
    else:
        t_min = []
    min_full = [np.argmin(np.abs(time - t)) for t in t_min]
    return d_nu, min_full


def plot_iteration(results, theory_params, threshold=0.05, full=True):
    traj = np.mean(results, axis=1)
    if full:
        ind = list(range(0, len(traj)))
    else:
        i = np.argmax(np.diff(traj[:, -1]))
        ind = [i, i+1]

    time = np.arange(0., theory_params['T'], theory_params['dt'])
    fig = pl.figure()
    ax = fig.add_subplot(121)
    [ax.plot(time,
             traj[i]) for i in ind]

    ax.set_yscale('Log')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel(r'$\langle \nu \rangle$')

    ax = fig.add_subplot(122)
    for n, i in enumerate(ind):
        d_nu, minima = velocity_peaks(time, results[i], threshold=threshold)
        ax.plot(time[:-1], d_nu)
        if not full:
            ax.vlines(time[minima],
                      1e-6,
                      1e0,
                      linestyles='dashed',
                      color='k')
    ax.set_yscale('Log')
    pl.show()

    
def save_iteration(step, data):
    data_dir = 'iteration_{}'.format(step)
    try:
        os.mkdir(data_dir)
    except FileExistsError:
        pass
    for key in ['parameters', 'K_prime', 'results']:
        np.save('iteration_{}/{}.npy'.format(step, key), data[key])


def load_iteration(step):
    data_dir = 'iteration_{}'.format(step)
    data = {}
    files = os.listdir(data_dir)
    for f in files:
        data[os.path.splitext(f)[0]] = np.load(os.path.join(data_dir, f))
    return data


def compute_iteration(max_iter, fac_nu_ext_5E_list, theory_params, M_base, threads=None):
    par_list = np.zeros(0)
    results = np.zeros((0, 254, int(theory_params['T'] / theory_params['dt'])))

    i = 0
    while i < max_iter:
        print("Iteration: {}".format(i))
        print(fac_nu_ext_5E_list)
        r = iteration_results(fac_nu_ext_5E_list, M_base, threads=threads)
        results = np.vstack((results, r))
        par_list = np.append(par_list, fac_nu_ext_5E_list)
        j = np.argmax(np.diff(np.mean(r, axis=1)[:, -1]))
        i += 1
        fac_nu_ext_5E_list = np.arange(fac_nu_ext_5E_list[j],
                                       # to ensure that the array
                                       # includes the last value, we
                                       # add a small epsilon
                                       fac_nu_ext_5E_list[j+1] + 1.e-10,
                                       10**(-(i+2.)))

    ind = np.argsort(par_list)
    par_list = par_list[ind]
    results = results[ind]

    return {'results': results, 'parameters': par_list}


def determine_velocity_minima(time, data, threshold=0.05):
    
    traj = np.mean(data['results'], axis=1)
    i = np.argmax(np.diff(traj[:, -1]))
    r_low = data['results'][i]
    r_high = data['results'][i+1]

    dnu_low, minima_low = velocity_peaks(time, r_low, threshold=threshold)
    dnu_high, minima_high = velocity_peaks(time, r_high, threshold=threshold)

    par_transition = data['parameters'][i]
    return par_transition, r_low, r_high, minima_low, minima_high

    

