# global imports
import numpy as np
from future.builtins import range

# local imports
import correlation_toolbox.helper as cthlp

'''
Documentation:

Correlation toolbox for AnalogSignals (binned data) of format

data = np.array([[t1,t2,...,tn], # unit1
                 [t1,t2,...,tn], # unit2
                 .
                 .
                 [t1,t2,...,tn]]) # unitN

Exception: compound_crossspec takes list of data, i.e. [data1,data2,...] as input.
'''


def mean(data, units=False, time=False):
    '''
    Compute mean of data

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       units: bool; average over units
       time: bool; average over time

    **Return**:
       if units=False and time=False: error, 
       if units=True: 1 dim numpy.ndarray; time series
       if time=True: 1 dim numpy.ndarray; series of unit means across time
       if units=True and time=True: float; unit and time mean

    **Examples**:
       >>> mean(np.array([[1,2,3],[4,5,6]]),units=True)
       Out[1]: np.array([2.5,3.5,4.5])

       >>> mean(np.array([[1,2,3],[4,5,6]]),time=True)
       Out[1]: np.array([2.,5.])

       >>> mean(np.array([[1,2,3],[4,5,6]]),units=True,time=True)
       Out[1]: 3.5

    '''

    assert(units is not False or time is not False)
    if units is True and time is False:
        return np.mean(data, axis=0)
    elif units is False and time is True:
        return np.mean(data, axis=1)
    elif units is True and time is True:
        return np.mean(data)


def compound_mean(data):
    '''
    Compute the mean of the compound/sum signal.
    data is first summed across units and averaged across time.

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time

    **Return**:
       float; time-averaged compound/sum signal
   
    **Examples**: 
       >>> compound_mean(np.array([[1,2,3],[4,5,6]]))
       Out[1]: 7.0

    '''

    return np.mean(np.sum(data, axis=0))


def variance(data, units=False, time=False):
    '''
    Compute the variance of data.

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       units: bool; variance across units
       time: bool; average over time

    **Return**:
       if units=False and time=False: error, 
       if units=True: 1 dim numpy.ndarray; time series,
       if time=True:  1 dim numpy.ndarray; series of single unit variances across time,
       if units=True and time=True: float; mean of single unit variances across time

    **Examples**: 
       >>> variance(np.array([[1,2,3],[4,5,6]]),units=True)
       Out[1]: np.array([ 2.25,  2.25,  2.25])
       >>> variance(np.array([[1,2,3],[4,5,6]]),time=True)
       Out[1]: np.array([ 0.66666667,  0.66666667])
       >>> variance(np.array([[1,2,3],[4,5,6]]),units=True,time=True)
       Out[1]: 0.66666666666666663

    '''

    assert(units is not False or time is not False)
    if units is True and time is False:
        return np.var(data, axis=0)
    elif units is False and time is True:
        return np.var(data, axis=1)
    elif units is True and time is True:
        return np.mean(np.var(data, axis=1))


def compound_variance(data):
    '''
    Compute the variance of the compound/sum signal.
    data is first summed across units, then the variance across time is calculated.
    
    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time

    **Return**:
       float; variance across time of compound/sum signal

    **Examples**: 
       >>> compound_variance(np.array([[1,2,3],[4,5,6]]))
       Out[1]: 2.6666666666666665

    '''

    return np.var(np.sum(data, axis=0))


def spectrogram(data, tbin, twindow, Df=None, units=False, N=None, measure='power'):
    '''Calculate (smoothed) spectrogram of data. If units is True, power
    spectra are averaged across units.

    Parameters:
    -----------
    data: binned timeseries
    tbin: bin size
    twindow: size of window use for spectra
    Df: width of smoothing kernel
    units: if True, average over units
    N: population size, if not given, calculated from data
    measure: define the measure to be used (power, cross, compound_power)
    '''

    steps_window = int(np.floor(twindow / tbin))
    n_windows = int(np.floor(1. * len(data[0]) / steps_window))
    sg = []
    freq = []
    if measure == 'power':
        for i in range(n_windows):
            freq, power = powerspec(data[:, i * steps_window:(i + 1) * steps_window], tbin, Df=Df, units=units, N=N)
            sg.append(power)
    elif measure == 'cross':
        for i in range(n_windows):
            freq, cross = crossspec(data[:, i * steps_window:(i + 1) * steps_window], tbin, Df=Df, units=units, N=N)
            sg.append(cross)
    elif measure == 'compound_power':
        for i in range(n_windows):
            freq, compound_power = compound_powerspec(data[:, i * steps_window:(i + 1) * steps_window], tbin, Df=Df)
            sg.append(compound_power)
    else:
        raise NotImplementedError('Unknow measure: %s.' % measure)

    return freq, np.array(sg)


def powerspec(data, tbin, Df=None, units=False, N=None):
    '''
    Calculate (smoothed) power spectra of all timeseries in data. 
    If units=True, power spectra are averaged across units.
    Note that averaging is done on power spectra rather than data.

    Power spectra are normalized by the length T of the time series -> no scaling with T. 
    For a Poisson process this yields:




    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       tbin: float; binsize in ms
       Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
       units: bool; average power spectrum 

    **Return**:
       (freq, POW): tuple
       freq: numpy.ndarray; frequencies
       POW: if units=False: 2 dim numpy.ndarray; 1st axis unit, 2nd axis frequency
            if units=True:  1 dim numpy.ndarray; frequency series

    **Examples**:
       >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df)
       Out[1]: (freq,POW)
       >>> POW.shape
       Out[2]: (2,len(analog_sig1))

       >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df, units=True)
       Out[1]: (freq,POW)
       >>> POW.shape
       Out[2]: (len(analog_sig1),)

    '''
    if N is None:
        N = len(data)
    freq, DATA = cthlp.calculate_fft(data, tbin)
    df = freq[1] - freq[0]
    T = tbin * len(freq)
    POW = np.power(np.abs(DATA),2)
    if Df is not None:
        POW = [cthlp.movav(x, Df, df) for x in POW]
        cut = int(Df / df)
        freq = freq[cut:]
        POW = np.array([x[cut:] for x in POW])
        POW = np.abs(POW)
    assert(len(freq) == len(POW[0]))
    if units is True:
        POW = 1./N*np.sum(POW, axis=0)
        assert(len(freq) == len(POW))
    POW *= 1. / T * 1e3  # normalization, power independent of T
    return freq, POW


def compound_powerspec(data, tbin, Df=None):
    '''
    Calculate the power spectrum of the compound/sum signal.
    data is first summed across units, then the power spectrum is calculated.

    Power spectrum is normalized by the length T of the time series -> no scaling with T. 
       
    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       tbin: float; binsize in ms
       Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
       
    **Return**:
       (freq, POW): tuple
       freq: numpy.ndarray; frequencies
       POW: 1 dim numpy.ndarray; frequency series

    **Examples**:
       >>> compound_powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df)
       Out[1]: (freq,POW)
       >>> POW.shape
       Out[2]: (len(analog_sig1),)

    '''

    return powerspec([np.sum(data, axis=0)], tbin, Df=Df, units=True)


def crossspec(data, tbin, Df=None, units=False, N=None):
    '''
    Calculate (smoothed) cross spectra of data.
    If units=True, cross spectra are averaged across units.
    Note that averaging is done on cross spectra rather than data.

    Cross spectra are normalized by the length T of the time series -> no scaling with T.

    Note that the average cross spectrum (units=True) is calculated efficiently via compound and single unit power spectra.
    
    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       tbin: float; binsize in ms
       Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
       units: bool; average cross spectrum

    **Return**:
       (freq, CRO): tuple
       freq: numpy.ndarray; frequencies
       CRO: if units=True:  1 dim numpy.ndarray; frequency series
            if units=False: 3 dim numpy.ndarray; 1st axis first unit, 2nd axis second unit, 3rd axis frequency

    **Examples**:
       >>> crossspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df)
       Out[1]: (freq,CRO)
       >>> CRO.shape
       Out[2]: (2,2,len(analog_sig1))

       >>> crossspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df, units=True)
       Out[1]: (freq,CRO)
       >>> CRO.shape
       Out[2]: (len(analog_sig1),)

    '''

    if N is None:
        N = len(data)
    if units is True:
        # smoothing and normalization take place in powerspec
        # and compound_powerspec
        freq, POW = powerspec(data, tbin, Df=Df, units=True, N=N)
        freq_com, CPOW = compound_powerspec(data, tbin, Df=Df)
        assert(len(freq) == len(freq_com))
        assert(np.min(freq) == np.min(freq_com))
        assert(np.max(freq) == np.max(freq_com))
        CRO = 1. / (1. * N * (N - 1.)) * (CPOW - 1. * N * POW)
        assert(len(freq) == len(CRO))
    else:
        freq, DATA = cthlp.calculate_fft(data, tbin)
        T = tbin * len(freq)
        df = freq[1] - freq[0]
        if Df is not None:
            cut = int(Df / df)
            freq = freq[cut:]
        CRO = np.zeros((N, N, len(freq)), dtype=complex)
        for i in range(N):
            for j in range(i + 1):
                tempij = DATA[i] * DATA[j].conj()
                if Df is not None:
                    tempij = cthlp.movav(tempij, Df, df)[cut:]
                CRO[i, j] = tempij
                CRO[j, i] = CRO[i, j].conj()
        assert(len(freq) == len(CRO[0, 0]))
        CRO *= 1. / T * 1e3  # normalization
    return freq, CRO


def compound_crossspec(a_data, tbin, Df=None):
    '''
    Calculate cross spectra of compound signals.
    a_data is a list of datasets (a_data = [data1,data2,...]). 
    For each dataset in a_data, the compound signal is calculated 
    and the crossspectra between these compound signals is computed.
           
    **Args**:
       a_data: list of numpy.ndarrays; array: 1st axis unit, 2nd axis time
       tbin: float; binsize in ms
       Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
       
    **Return**:
       (freq, CRO): tuple
       freq: numpy.ndarray; frequencies
       CRO: 3 dim numpy.ndarray; 1st axis first compound signal, 2nd axis second compound signal, 3rd axis frequency

    **Examples**:
       >>> compound_crossspec([np.array([analog_sig1,analog_sig2]),np.array([analog_sig3,analog_sig4])],tbin, Df=Df)
       Out[1]: (freq,CRO)
       >>> CRO.shape
       Out[2]: (2,2,len(analog_sig1))

    '''

    a_mdata = []
    for data in a_data:
        a_mdata.append(np.sum(data, axis=0))  # calculate compound signals
    return crossspec(np.array(a_mdata), tbin, Df, units=False)


def autocorrfunc(freq, power):
    '''
    Calculate autocorrelation function(s) for given power spectrum/spectra.
 
    For a Poisson process this yields:

    **Args**:
       freq: 1 dim numpy.ndarray; frequencies 
       power: 2 dim numpy.ndarray; power spectra, 1st axis units, 2nd axis frequencies

    **Return**:
       (time,autof): tuple
       time: 1 dim numpy.ndarray; times
       autof: 2 dim numpy.ndarray; autocorrelation functions, 1st axis units, 2nd axis times

    **Examples**:
       ---

    '''
    tbin = 1. / (2. * np.max(freq)) * 1e3  # tbin in ms
    time = np.arange(-len(freq) / 2. + 1, len(freq) / 2. + 1) * tbin
    # T = max(time)
    multidata = False
    if len(np.shape(power)) > 1:
        multidata = True
    if multidata:
        N = len(power)
        autof = np.zeros((N, len(freq)))
        for i in range(N):
            raw_autof = np.real(np.fft.ifft(power[i]))
            mid = int(len(raw_autof) / 2.)
            autof[i] = np.hstack([raw_autof[mid + 1:], raw_autof[:mid + 1]])
        assert(len(time) == len(autof[0]))
    else:
        raw_autof = np.real(np.fft.ifft(power))
        mid = int(len(raw_autof) / 2.)
        autof = np.hstack([raw_autof[mid + 1:], raw_autof[:mid + 1]])
        assert(len(time) == len(autof))
    #autof *= T*1e-3 # normalization is done in powerspec()
    return time, autof


def autocorrfunc_time(spike_trains, tau_max, bin_size, T, units=False):
    '''
    Calculate autocorrelation function(s) for given spike trains.

    **Args**:
       spike_trains: 2 dim numpy.ndarray; 1st axis units, 2nd axis spike times
       tau_max: maximal time-lag of correlation function

    **Return**:
       (time,autof): tuple
       time: 1 dim numpy.ndarray; times
       autof: 2 dim numpy.ndarray; autocorrelation functions, 1st axis units, 2nd axis times

    '''

    if 2*tau_max >= T :
        raise RuntimeError('tau_max has to be smaller than T/2')
    nr_units = len(spike_trains)
    spike_trains = [np.asarray(st) for st in spike_trains]
    # adjust tau_max such that tau_max-bin_size/2 is a multiple of the bin_size 
    N = int(tau_max/bin_size-0.5)
    tau_max = N * bin_size + bin_size/2.
    nr_bins = 2*N + 1
    if units == False:
        auto = np.zeros((nr_units,nr_bins))
    else:
        auto = np.zeros(nr_bins)
    # remove the time intervall tau_max from the beginning and end of the reference spike train, to avoid edge effects due to
    # finiteness of spike train
    trimmed_spikes = [st[np.where(st>tau_max)] for st in spike_trains]
    trimmed_spikes = [ts[np.where(ts<T-tau_max)] for ts in trimmed_spikes]
    # loop of spike trains
    for i,ts in enumerate(trimmed_spikes):
        # loop over spikes in one spike train
        for spike in ts:
            # get difference time difference between this spike and all other spikes
            diff = spike_trains[i]-spike
            diff = diff[np.where(abs(diff)<=tau_max)]
            # find correct bin
            diff = (tau_max+diff)/bin_size
            diff = diff.astype(int)
            if units == False:
                auto[i][diff] += 1
            else:
                auto[diff] += 1/float(nr_units)
    auto = auto * 1000.0
    t_start = -tau_max + bin_size/2.
    t_end = tau_max - bin_size/2.
    time = np.arange(t_start,t_end+bin_size,bin_size)
    auto /= (T-2*tau_max)
    return time, auto


def crosscorrfunc(freq, cross):
    '''
    Calculate crosscorrelation function(s) for given cross spectra.

    **Args**:
       freq: 1 dim numpy.ndarray; frequencies 
       cross: 3 dim numpy.ndarray; cross spectra, 1st axis units, 2nd axis units, 3rd axis frequencies

    **Return**:
       (time,crossf): tuple
       time: 1 dim numpy.ndarray; times
       crossf: 3 dim numpy.ndarray; crosscorrelation functions, 1st axis first unit, 2nd axis second unit, 3rd axis times

    **Examples**: 
       ---

    '''

    tbin = 1. / (2. * np.max(freq)) * 1e3  # tbin in ms
    time = np.arange(-len(freq) / 2. + 1, len(freq) / 2. + 1) * tbin
    # T = max(time)
    multidata = False
    # check whether cross contains many cross spectra
    if len(np.shape(cross)) > 1:
        multidata = True
    if multidata:
        N = len(cross)
        crossf = np.zeros((N, N, len(freq)))
        for i in range(N):
            for j in range(N):
                raw_crossf = np.real(np.fft.ifft(cross[i, j]))
                mid = int(len(raw_crossf) / 2.)
                crossf[i, j] = np.hstack(
                    [raw_crossf[mid + 1:], raw_crossf[:mid + 1]])
        assert(len(time) == len(crossf[0, 0]))
    else:
        raw_crossf = np.real(np.fft.ifft(cross))
        mid = int(len(raw_crossf) / 2.)
        crossf = np.hstack([raw_crossf[mid + 1:], raw_crossf[:mid + 1]])
        assert(len(time) == len(crossf))
    # crossf *= T*1e-3 # normalization happens in cross spectrum
    return time, crossf


def corrcoef(time, crossf, integration_window=0.):
    '''
    Calculate the correlation coefficient for given auto- and crosscorrelation functions. 
    Standard settings yield the zero lag correlation coefficient.
    Setting integration_window > 0 yields the correlation coefficient of integrated auto- and crosscorrelation functions.
    The correlation coefficient between a zero signal with any other signal is defined as 0.

    \begin{equation}
    corrcoeff_{1,2} = \frac{crossf_{1,2}}{\sqrt{autof_1*autof_2}}
    \end{equation}

    **Args**:
       time: 1 dim numpy.ndarray; times corresponding to signal
       crossf: 3 dim numpy.ndarray; crosscorrelation functions, 1st axis first unit, 2nd axis second unit, 3rd axis times
       integration_window: float; 

    **Return**:
       cc: 2 dim numpy.ndarray; correlation coefficient between two units

    **Examples**: 
       ---

    '''

    N = len(crossf)
    cc = np.zeros(np.shape(crossf)[:-1])
    tbin = abs(time[1] - time[0])
    lim = int(integration_window / tbin)
    if len(time)%2 == 0:
        mid = int(len(time)/2-1)
    else:
        mid = int(np.floor(len(time)/2.))
    for i in range(N):

        ai = np.sum(crossf[i, i][mid - lim:mid + lim + 1])
        offset_autoi = np.mean(crossf[i,i][:mid-1])
        for j in range(N):
            cij = np.sum(crossf[i, j][mid - lim:mid + lim + 1])
            offset_cross = np.mean(crossf[i,j][:mid-1])
            aj = np.sum(crossf[j, j][mid - lim:mid + lim + 1])
            offset_autoj = np.mean(crossf[j,j][:mid-1])
            if ai > 0. and aj > 0.:
                cc[i, j] = (cij-offset_cross) / np.sqrt((ai-offset_autoi) * (aj-offset_autoj))
            else:
                cc[i, j] = 0.
    return cc


def coherence(freq, power, freq_cross, cross):
    '''
    Calculate frequency resolved complex coherence for given power- and crossspectra.

    \begin{equation}
    coherence_{1,2} = \frac{crossspec_{1,2}}{\sqrt{powerspec_1*powerspec_2}}
    \end{equation}

    **Args**:
       freq: 1 dim numpy.ndarray; frequencies
       power: 2 dim numpy.ndarray; power spectra, 1st axis units, 2nd axis frequencies
       cross: 3 dim numpy.ndarray; cross spectra, 1st axis units, 2nd axis units, 3rd axis frequencies

    **Return**:
       (freq,coh): tuple
       freq: 1 dim numpy.ndarray; frequencies
       coh: 3 dim numpy.ndarray; coherences, 1st axis units, 2nd axis units, 3rd axis frequencies

    **Examples**: 
       ---

    '''

    assert(min(freq) == min(freq_cross))
    assert(max(freq) == max(freq_cross))
    df = freq[1]-freq[0]
    df_cross = freq_cross[1]-freq_cross[0]
    assert(df == df_cross)
    if len(np.shape(cross)) > 1:
        N = len(power)
        coh = np.zeros_like(cross)
        for i in range(N):
            for j in range(N):
                coh[i, j] = cross[i, j] / np.sqrt(power[i] * power[j])
        assert(len(freq) == len(coh[0, 0]))
    else:
        coh = cross / power
    return freq, coh


def cv(data, units=False):
    '''
    Calculate coefficient of variation for data. Mean and standard deviation are computed across time.

    \begin{equation}
    CV = \frac{\sigma}{\mu}
    \end{equation}

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       units: bool; average CV

    **Return**:
       if units=False: numpy.ndarray; series of unit CVs
       if units=True: float; mean CV across units      

    **Examples**:
       >>> cv(np.array([[1,2,3,4,5,6],[11,2,3,3,4,5]]))
       Out[1]: np.array([ 0.48795004,  0.63887656])

       >>> cv(np.array([[1,2,3,4,5,6],[11,2,3,3,4,5]]),units=True)
       Out[1]: 0.56341330073710316
 
    '''

    mu = mean(data, time=True)
    var = variance(data, time=True)
    cv = np.sqrt(var) / mu
    if units is True:
        return np.mean(cv)
    else:
        return cv


def fano(data, units=False):
    '''
    Calculate fano factor for data. Mean and variance are computed across time.

    \begin{equation}
    FF = \frac{\sigma^2}{\mu}
    \end{equation}

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       units: bool; average FF

    **Return**:
       if units=False: numpy.ndarray; series of unit FFs
       if units=True: float; mean FF across units      

    **Examples**       
       >>> fano(np.array([[1,2,3,4,5,6],[11,2,3,3,4,5]]))
       Out[1]: np.array([0.83333333, 1.9047619])

       >>> fano(np.array([[1,2,3,4,5,6],[11,2,3,3,4,5]]),units=True)
       Out[1]: 1.3690476190476191

    '''

    mu = mean(data, time=True)
    var = variance(data, time=True)
    ff = var / mu
    if units is True:
        return np.mean(ff)
    else:
        return ff
