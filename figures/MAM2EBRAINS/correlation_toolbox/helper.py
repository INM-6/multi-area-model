# global imports
import numpy as np
import copy
from future.builtins import range

##################################################################
# functions for handling spike train data
##################################################################

# TODO create continous process with given correlation (e.g. Gauss process)


def create_poisson_spiketrains(rate, T, N):
    '''
    returns an array (gdf) containing indices of N neurons
    and corresponding spiketimes with [number of spikes]/neuron/s=rate
    and poissonian ISI distribution
    returns gdf: [(id,time),...]
    '''
    N = int(N)
    times = np.sort(np.hstack(
        [T * np.random.uniform(size=np.random.poisson(rate * T * 1e-3))
         for _ in range(N)]))
    ids = np.random.random_integers(0, N - 1, len(times))
    return np.array(list(zip(ids, times)))


def create_gamma_spiketrains(rate, T, N, k):
    N = int(N)
    size = int(1.5 * rate * T)
    times = np.hstack([np.cumsum([np.random.gamma(rate, k, size=size)])
                      for i in range(N)])
    ids = np.hstack([[i] * size for i in range(N)])
    keep = np.where(times < T)
    times = times[keep]
    ids = ids[keep]
    srt = np.argsort(times)
    times = times[srt]
    ids = ids[srt]
    return np.array(list(zip(ids, times)))


def sort_gdf_by_id(data, idmin=None, idmax=None):
    '''
    Sort gdf data [(id,time),...] by neuron id.

    Parameters
    ----------

    data: numpy.array (dtype=object) with lists ['int', 'float']
          The nest output loaded from gdf format. Each row contains a global id
    idmin, idmax : int, optional
            The minimum/maximum neuron id to be considered.

    Returns
    -------
    ids : list of ints
          Neuron ids, e.g., [id1,id2,...]
    srt : list of lists of floats
          Spike trains corresponding to the neuron ids, e.g., [[t1,t2,...],...]
    '''

    assert((idmin is None and idmax is None)
           or (idmin is not None and idmax is not None))

    # get neuron ids
    if idmin is None and idmax is None:
        ids = np.unique(data[:, 0])
    else:
        ids = np.arange(idmin, idmax+1)
    srt = []
    for i in ids:
        srt.append(np.sort(data[np.where(data[:, 0] == i)[0], 1]))
    if len(ids) == 0:
        print('CT warning(sort_spiketrains_by_id): empty gdf data!')
    return ids, srt


def gdf_to_neo(data,
               quantity,
               block,
               index,
               idmin=None,
               idmax=None,
               t_start=None,
               t_stop=None):
    '''
    Take spike trains from NEST output in gdf format, convert them to neo
    segments and append them to a given block structure.

    Parameters
    ----------

    data : numpy.array (dtype=object) with lists ['int', 'float']
           The nest output loaded from gdf format. Each row contains a global id
           of a neuron and a spike time.
    quantity : quantity
               The unit of the spike times, e.g. quantities.ms
    block : neo.Block
            A container to which the spike trains should be appended.
    index : int
            The index of the segment to which to append the spike trains.
    idmin, idmax : int, optional
            The minimum/maximum neuron id to be considered.
    t_start, t_stop : float, optional
              The minimum/maximum spike time to be considered.

    Returns
    -------
    block : neo.Block
            The container given as input argument extended by the spike trains
            in data.
    '''

    import neo

    assert((idmin is None and idmax is None)
           or (idmin is not None and idmax is not None))

    if len(data) > 0:
        # sort data by neuron ids
        ids, srt = sort_gdf_by_id(data, idmin=idmin, idmax=idmax)
        # go through all segments of the block and check whether any segment
        # has the given index
        for seg in block.segments:
            if seg.index == index:
                # append the spike trains to the existing segment
                for i, idx in enumerate(ids):
                    seg.spiketrains.append(
                        neo.core.SpikeTrain(srt[i]*quantity,
                                            t_start=t_start,
                                            t_stop=t_stop,
                                            annotations={'unit_id': idx}))
                return block
        # if a segment of the given index does not exist, yet, create it
        seg = neo.core.Segment(index=index)
        for i, idx in enumerate(ids):
            seg.spiketrains.append(
                neo.core.SpikeTrain(srt[i]*quantity,
                                    t_start=t_start,
                                    t_stop=t_stop,
                                    annotations={'unit_id': idx}))
        block.segments.append(seg)
        return block
    else:
        print('CT warning(sort_spiketrains_by_id): empty gdf data!')
        return


def sort_membrane_by_id(data, idmin=None, idmax=None):
    '''
    sort recorded membrane potentials
    [id, time, v] by neuron id
    assumes time steps are equal for all traces
    '''
    if len(data) > 0:
        if idmin is None and idmax is None:
            ids = np.unique(data[0:, 0])
        else:
            ids = np.arange(idmin, idmax)
        srt = []
        for i in ids:
            srt.append(data[np.where(data[0:, 0] == i)[0], 2])
        tim = np.sort(data[np.where(data[0:, 0] == ids[0])[0], 1])
        return ids, tim, srt
    else:
        print('CT warning(sort_membrane_by_id): empty membrane data!')
        return None, None, None


def instantaneous_spike_count(data, tbin, tmin=None, tmax=None):
    '''
    Create a histogram of spike trains
    returns bins, hist
    '''
    if tmin is None:
        tmin = np.min([np.min(x) for x in data if len(x) > 0])
    if tmax is None:
        tmax = np.max([np.max(x) for x in data if len(x) > 0])
    assert(tmin < tmax)
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist


def instantaneous_firing_rate(data, tbin, tmin=None, tmax=None):
    '''
    Create a histogram for spike trains, taking into account bin width
    '''
    bins, hist = instantaneous_spike_count(data, tbin, tmin, tmax)
    return bins, hist * 1. / tbin * 1e3


def strip_sorted_spiketrains(sp):
    '''
    removes sorted spiketrains which do not contain a single spike
    '''
    return np.array([x for x in sp if len(x) > 0])


def strip_binned_spiketrains(sp):
    '''
    removes binned spiketrains which do not contain a single spike
    '''
    return np.array([x for x in sp if abs(np.max(x) - np.min(x)) > 1e-16])


def create_correlated_spiketrains_sip(rate, T, N, cc):
    '''
    create N correlated spiketrains (SIP) with rate rate,
    duration T and corrcoef cc
    returns gdf: [(id,time),...]
    '''
    rated = (1. - cc) * rate  # rate of independent (disjoint) processes
    ratec = cc * rate  # rate of common process
    d = create_poisson_spiketrains(rated, T, N)
    c = create_poisson_spiketrains(ratec, T, 1)
    for i in np.unique(d[:, 0]):
        c[:, 0] = [i] * len(c)
        d = np.vstack([d, c])
    return d[np.argsort(d[:, 1])]


##################################################################
# general helper functions
##################################################################

def movav(y, Dx, dx):
    '''
    calculate average of signal y by using sliding rectangular
    window of size Dx using binsize dx
    '''
    if Dx <= dx:
        return y
    else:
        ly = len(y)
        r = np.zeros(ly)
        n = np.int(np.round((Dx / dx)))
        r[0:np.int(n / 2.)] = 1.0 / n
        r[-np.int(n / 2.)::] = 1.0 / n
        R = np.fft.fft(r)
        Y = np.fft.fft(y)
        yf = np.fft.ifft(Y * R)
        return yf


def calculate_fft(data, tbin):
    '''
    calculate the fouriertransform of data
    [tbin] = ms
    '''
    if len(np.shape(data)) > 1:
        n = len(data[0])
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data, axis=1)
    else:
        n = len(data)
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data)


def centralize(data, time=False, units=False):
    assert(time is not False or units is not False)
    res = copy.copy(data)
    if time is True:
        res = np.array([x - np.mean(x) for x in res])
    if units is True:
        res = np.array(res - np.mean(res, axis=0))
    return res
