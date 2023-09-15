import numpy as np
import subprocess
import matplotlib.pyplot as plt

from multiarea_model import Analysis
from config import data_path

def load_and_create_data(M, A, raster_areas):
    """
    Calculate time-averaged population rates and store them in member pop_rates.
    If the rates had previously been stored with the same
    parameters, they are loaded from file.

    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    compute_stat : bool, optional
        If set to true, the mean and variance of the population rate
        is calculated. Defaults to False.
        Caution: Setting to True slows down the computation.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    """
    A.create_pop_rates()
    # subprocess.run(['python3', './figures/Schmidt2018_dyn/compute_pop_rates.py'])
    # subprocess.run(['Rscript', '--vanilla', 'compute_bold_signal.R', fn, out_fn])
    # print("Computing population rates done")

    
    """
    Calculate poulation-averaged LvR (see Shinomoto et al. 2009) and
    store as member pop_LvR. Uses helper function LvR.

    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    """
    A.create_pop_LvR()
    # print("Computing population LvR done")
    
    
    """
    Calculate time series of population- and area-averaged firing rates.
    Uses ah.pop_rate_time_series.
    If the rates have previously been stored with the
    same parameters, they are loaded from file.


    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
    """
    A.create_rate_time_series()
    # print("Computing rate time series done")
    
    
    # # Compute time series of firing rates convolved with a kernel
    # # data_path = sys.argv[1]
    # print(data_path)
    # label = M.simulation.label
    # method = 'auto_kernel'
    # for area in raster_areas: 
    #     subprocess.run(['python3', './figures/Schmidt2018_dyn/compute_rate_time_series.py', data_path, label, area, method])
    # print("Computing rate time series auto kernel done")
    
    
    """
    Calculate synchrony as the coefficient of variation of the population rate
    and store in member synchrony. Uses helper function synchrony.
    If the synchrony has previously been stored with the
    same parameters, they are loaded from file.


    Parameters
    ----------
    t_min : float, optional
        Minimal time in ms of the simulation to take into account
        for the calculation. Defaults to 500 ms.
    t_max : float, optional
        Maximal time in ms of the simulation to take into account
        for the calculation. Defaults to the simulation time.
    areas : list, optional
        Which areas to include in the calculcation.
        Defaults to all loaded areas.
    pops : list or {'complete'}, optional
        Which populations to include in the calculation.
        If set to 'complete', all populations the respective areas
        are included. Defaults to 'complete'.
    resolution : float, optional
        Resolution of the population rate. Defaults to 1 ms.
    """
    A.create_synchrony()
    print("Computing synchrony done")
    
    
#     # Create corrrelation coefficient
#     # data_path = data_path
#     label = M.simulation.label

#     # Run the script with arguments
#     subprocess.run(['python', './figures/Schmidt2018_dyn/compute_corrcoeff.py', data_path, label])
    
    
#     """
#     Calculate synaptic input of populations and areas using the spike data.
#     Uses function ah.pop_synaptic_input.
#     If the synaptic inputs have previously been stored with the
#     same parameters, they are loaded from file.

#     Parameters
#     ----------
#     t_min : float, optional
#         Minimal time in ms of the simulation to take into account
#         for the calculation. Defaults to 500 ms.
#     t_max : float, optional
#         Maximal time in ms of the simulation to take into account
#         for the calculation. Defaults to the simulation time.
#     areas : list, optional
#         Which areas to include in the calculcation.
#         Defaults to all loaded areas.
#     pops : list or {'complete'}, optional
#         Which populations to include in the calculation.
#         If set to 'complete', all populations the respective areas
#         are included. Defaults to 'complete'.
#     kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
#         Convolution kernel for the calculation of the underlying firing rates.
#         Defaults to 'binned' which corresponds to a simple histogram.
#     resolution: float, optional
#         Width of the convolution kernel. Specifically it correponds to:
#         - 'binned' : bin width of the histogram
#         - 'gauss_time_window' : sigma
#         - 'alpha_time_window' : time constant of the alpha function
#         - 'rect_time_window' : width of the moving rectangular function
#     """
    # A.create_synaptic_input()
    # # print("Computing synaptic input done")
    
    A.save()
    
    # """
    # Compute BOLD signal for a given area from the time series of
    # population-averaged spike rates of a given simulation using the
    # neuRosim package of R (see Schmidt et al. 2018 for more details).
    # """
    # try:
    #     subprocess.run(['python3', './Schmidt2018_dyn/compute_bold_signal.py'])
    # except FileNotFoundError:
    #     raise FileNotFoundError("Executing R failed. Did you install R?")