"""
theory_helpers
============

Helper function for the theory class.
Evaluates the Siegert formula and its
derivatives.


Functions
--------


Authors
--------
Maximilian Schmidt
Jannis Schuecker
Moritz Helias

"""

import numpy as np
import scipy
import scipy.integrate
import scipy.stats
import scipy.special


def nu0_fb(mu, sigma, tau_m, tau_s, tau_r, V_th, V_r):
    """
    Compute the stationary firing rate of a neuron with synaptic
    filter of time constant tau_s driven by Gaussian white noise, from
    Fourcaud & Brunel 2002.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_s : float
        Synaptic time constant of the neuron in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """
    alpha = np.sqrt(2) * abs(scipy.special.zetac(0.5) + 1)

    # effective threshold
    V_th1 = V_th + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    # effective reset
    V_r1 = V_r + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # use standard Siegert with modified threshold and reset
    return nu_0(tau_m, tau_r, V_th1, V_r1, mu, sigma)


def nu_0(tau_m, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the stationary firing rate of neuron
    without synaptic filtering.
    Evaluate the Siegert function given the
    mean and variance of the input current.
    See Eq. 3 of Schuecker, Schmidt et al. (2017).
    This function decides automatically whether to use
    siegert1 or siegert2.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """
    if mu <= V_th - 0.05 * abs(V_th):
        return siegert1(tau_m, tau_r, V_th, V_r, mu, sigma)
    else:
        return siegert2(tau_m, tau_r, V_th, V_r, mu, sigma)


def siegert1(tau_m, tau_r, V_th, V_r, mu, sigma):
    """
    Evaluate the Siegert function given the
    mean and variance of the input current.
    See Eq. 3 of Schuecker, Schmidt et al. (2017).
    Use this function if the mean input current is
    below the threshold potential.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    y_th = (V_th - mu) / sigma
    y_r = (V_r - mu) / sigma

    def integrand(u):
        if u == 0:
            return np.exp(-y_th**2) * 2 * (y_th - y_r)
        else:
            return np.exp(-(u - y_th)**2) * (1.0 - np.exp(2 * (y_r - y_th) * u)) / u

    lower_bound = y_th
    err_dn = 1.0
    while err_dn > 1e-12 and lower_bound > 1e-16:
        err_dn = integrand(lower_bound)
        if err_dn > 1e-12:
            lower_bound /= 2

    upper_bound = y_th
    err_up = 1.0
    while err_up > 1e-12:
        err_up = integrand(upper_bound)
        if err_up > 1e-12:
            upper_bound *= 2

    # Check to prevent overflow:
    if y_th >= 20:
        out = 0.
    if y_th < 20:
        out = 1.0 / (tau_r + np.exp(y_th**2) *
                     scipy.integrate.quad(integrand, lower_bound,
                                          upper_bound)[0] * tau_m)
    else:
        out = 0.
    return out


def siegert2(tau_m, tau_r, V_th, V_r, mu, sigma):
    """
    Evaluate the Siegert function given the
    mean and variance of the input current.
    See Eq. 3 of Schuecker, Schmidt et al. (2017).
    Use this function if the mean input current is
    above the threshold potential.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    y_th = (V_th - mu) / sigma
    y_r = (V_r - mu) / sigma

    def integrand(u):
        if u == 0:
            return 2 * (y_th - y_r)
        else:
            return (np.exp(2 * y_th * u - u**2) - np.exp(2 * y_r * u - u**2)) / u

    upper_bound = 1.0
    err = 1.0
    while err > 1e-12:
        err = integrand(upper_bound)
        upper_bound *= 2

    return 1.0 / (tau_r + scipy.integrate.quad(integrand, 0.0, upper_bound)[0] * tau_m)


def d_nu_d_mu_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the derivative of the firing rate with respect
    to the mean of the input current for a neuron with
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_s : float
        Synaptic time constant of the neuron in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    alpha = np.sqrt(2) * abs(scipy.special.zetac(0.5) + 1)

    # effective threshold
    V_th1 = V_th + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    # effective reset
    V_r1 = V_r + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    return d_nu_d_mu_numeric(tau_m, tau_r, V_th1, V_r1, mu, sigma)


def d_nu_d_mu_numeric(tau_m, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the derivative of the firing rate with respect
    to the mean of the input current for a neuron without
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    eps = 0.01
    nu0_minus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    nu0_plus = nu_0(tau_m, tau_r, V_th, V_r, mu + eps, sigma)

    return (nu0_plus - nu0_minus) / eps


def d_nu_d_sigma_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the derivative of the firing rate with respect
    to the variance of the input current for a neuron with
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_s : float
        Synaptic time constant of the neuron in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    alpha = np.sqrt(2) * abs(scipy.special.zetac(0.5) + 1)

    # effective threshold
    V_th1 = V_th + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    # effective reset
    V_r1 = V_r + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    return d_nu_d_sigma_numeric(tau_m, tau_r, V_th1, V_r1, mu, sigma)


def d2_nu_d_sigma_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the 2nd derivative of the firing rate with respect
    to the variance of the input current for a neuron with
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_s : float
        Synaptic time constant of the neuron in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    eps = 0.01
    sigma0_minus = d_nu_d_sigma_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma)
    sigma0_plus = d_nu_d_sigma_fb_numeric(
        tau_m, tau_s, tau_r, V_th, V_r, mu, sigma + eps)
    return (sigma0_plus - sigma0_minus) / eps


def d_nu_d_sigma_numeric(tau_m, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the 2nd derivative of the firing rate with respect
    to the variance of the input current for a neuron without
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    eps = 0.01
    nu0_minus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    nu0_plus = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma + eps)

    return (nu0_plus - nu0_minus) / eps


def d2_nu_d_mu_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """
    Compute the 2nd derivative of the firing rate with respect
    to the mean of the input current for a neuron with
    synaptic filtering.

    Parameters
    ----------
    tau_m : float
        Membrane time constant of the neurons in ms.
    tau_s : float
        Synaptic time constant of the neuron in ms.
    tau_r : float
        Refractory time of the neurons in ms.
    V_th : float
        Threshold membrane potential of the neurons in mV.
    V_r : float
        Reset potential of the neurons in mV.
    mu : float
        Mean of the input current to the neurons in mV
    sigma : float
        Variance of the input current to the neurons in mV
    """

    eps = 0.01
    nu0_minus = d_nu_d_mu_fb_numeric(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma)
    nu0_plus = d_nu_d_mu_fb_numeric(
        tau_m, tau_s, tau_r, V_th, V_r, mu + eps, sigma)
    return (nu0_plus - nu0_minus) / eps
