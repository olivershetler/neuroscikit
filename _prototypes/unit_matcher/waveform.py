"""
Functions that take a single waveform as an input and return a feature.
"""

import numpy as np
from operator import add

def waveform_features(waveform, delta):
    """
    Calculate the features of a waveform.
    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.

    Returns
    -------
    feature vector
        A vector of features of the waveform.

    References:
    ----------
    Caro-Martín, Carmen Rocío, José M. Delgado-García, Agnès Gruart, and R. Sánchez-Campusano. “Spike Sorting Based on Shape, Phase, and Distribution Features, and K-TOPS Clustering with Validity and Error Indices.” Scientific Reports 8, no. 1 (December 12, 2018): 17796. https://doi.org/10.1038/s41598-018-35491-4.
    """
    # get domains
    t = time_index(waveform, delta)
    d_waveform = derivative(waveform, delta)
    d2_waveform = derivative2(waveform, delta)
    # get morphological points
    p1, p2, p3, p4, p5, p6 = morphological_points(t, waveform, d_waveform, d2_waveform, delta)

    # FEATURE EXTRACTION
    fd = dict() # feature dictionary

    # get peak amplitude for source attribution
    fd["peak_amplitude"] = p3.v

    # SHAPE FEATURES
    # waveform duration of the first derivative (FD) of the action potential (AP)
    fd["f1"] = p5.t - p1.t
    # peak to valley amplitude of the FD of the AP
    fd["f2"] = p4.dv - p2.dv
    # valley to valley amplitude of the FD of the AP
    fd["f3"] = p6.dv - p2.dv
    # integral of the spike slice in the waveform, normalized for time
    # NOTE: This feature is NOT in the original paper
    # in the original paper, f4 is the correlation between
    # the waveform and a reference waveform (we don't use reference waveforms).
    fd["f4"] = area_under_curve(waveform[p1.i:p5.i], delta)/(p5.t - p1.t)
    # logarithm of the positve deflection of the FD of the AP
    # NOTE: This feature is NOT in the original paper
    # in the original paper, f5 is the logarithm of the term below
    # However, their definition did not generalize to excidatory
    # neurons, where the principal peak comes before the big trough.
    fd["f5"] = symmetric_logarithm((p4.dv - p2.dv) / (p4.t - p2.t))
    # negative deflection of the FD of the AP
    fd["f6"] = (p6.dv - p4.dv) / (p6.t - p4.t)
    # logarithm of the slope among valleys of the FD of the AP
    fd["f7"] = symmetric_logarithm((p6.dv - p2.dv) / (p6.t - p2.t))
    # root mean square of the pre-event amplitude of the FD of the AP
    # NOTE: This feature is MODIFIED from the original paper
    # in the original paper, f8 is the root mean square of the pre-event amplitude of the FD of the AP
    # However, their definition did not generalize to excidatory
    # neurons, where the first extremum before the principal peak could
    # be the boundary of the pre-event amplitude.
    # We use the first extremum of the first derivative as the cutoff
    # when the first voltage domain extremum is the boundary.
    fd["f8"] = np.sqrt(np.mean([x**2 for x in d_waveform[:p1.i]])) if p1.i > 0 else np.sqrt(np.mean([x**2 for x in d_waveform[p1.i:p2.i]]))
    # negative slope ratio of the FD of the AP
    fd["f9"] = ((p2.dv - p1.dv)/(p2.t - p1.t))/((p3.dv - p2.dv)/(p3.t - p2.t))
    # postive slope ratio of the FD of the AP
    fd["f10"] = ((p4.dv - p3.dv)/(p4.t - p3.t))/((p5.dv - p4.dv)/(p5.t - p4.t))
    # peak to valley ratio of the action potential
    fd["f11"] = p2.dv / p4.dv
    # PHASE FEATURES
    # amplitude of the FD of the AP relating to p1
    fd["f12"] = p1.dv
    # amplitude of the FD of the AP relating to p3
    fd["f13"] = p3.dv
    # amplitude of the FD of the AP relating to p4
    fd["f14"] = p4.dv
    # amplitude of the FD of the AP relating to p6
    fd["f15"] = p5.dv
    # amplitude of the FD of the AP relating to p6
    fd["f16"] = p6.dv
    # amplitude of the second derivative (SD) of the AP relating to p1
    fd["f17"] = p1.d2v
    # amplitude of the SD of the AP relating to p3
    fd["f18"] = p3.d2v
    # amplitude of the SD of the AP relating to p5
    fd["f19"] = p5.d2v
    # inter-quartile range of the FD of the AP
    fd["f20"] = inter_quartile_range(d_waveform)
    # inter-quartile range of the SD of the AP
    fd["f21"] = inter_quartile_range(d2_waveform)
    # kurtosis coefficient of the FD of the AP
    fd["f22"] = kurtosis(d_waveform)
    # skew / Fisher assymetry of the FD of the AP
    fd["f23"] = skew(d_waveform)
    # skew / Fisher assymetry of the SD of the AP
    fd["f24"] = skew(d2_waveform)

    for key, value in fd.items():
        fd[key] = float(value)

    return fd

def morphological_points(time_index, waveform, d_waveform, d2_waveform, delta):
    """
    Find the key morphological points in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.
    d2_waveform : array_like
        The second derivative of the waveform to be analyzed.

    Returns
    -------
    p1, p2, p3, p4, p5, p6 : tuple
        The key morphological points in the waveform.

    References:
    ----------
    Caro-Martín, Carmen Rocío, José M. Delgado-García, Agnès Gruart, and R. Sánchez-Campusano. “Spike Sorting Based on Shape, Phase, and Distribution Features, and K-TOPS Clustering with Validity and Error Indices.” Scientific Reports 8, no. 1 (December 12, 2018): 17796. https://doi.org/10.1038/s41598-018-35491-4.
    """

    waveform_point = lambda i: Point(i, time_index, waveform, d_waveform, d2_waveform)

    # get morphological points in the voltage domain
    voltage_extrema_indexes = local_extrema(waveform, delta)
    voltage_extrema_values = [waveform[i] for i in voltage_extrema_indexes]
    x = int(np.argmax(voltage_extrema_values))
    # find principal voltage peak
    p3 = waveform_point(voltage_extrema_indexes[x])
    # get pre-spike trough
    p1 = waveform_point(voltage_extrema_indexes[x - 1])
    # get refractory trough
    p5 = waveform_point(voltage_extrema_indexes[x + 1])
    # get refractory peak index (discard after use)
    rp = waveform_point(voltage_extrema_indexes[x + 2])

    # get morphological points in the rate domain
    def steepest_point_in_region(start, end):
        rate_extrema_indexes = local_extrema(d_waveform, delta)
        r = lambda start, end: list(filter(lambda i: i >= start.i and i <= end.i, rate_extrema_indexes))
        v = lambda indexes: [abs(d_waveform[i]) for i in indexes]
        indexes = r(start, end)
        x = np.argmax(v(indexes))
        return waveform_point(indexes[x])
    # get steepest point between pre-spike trough and principal peak
    p2 = steepest_point_in_region(p1, p3)
    # get steepest point between principal peak and refractory trough
    p4 = steepest_point_in_region(p3, p5)
    # get steepest point between refractory trough and refractory peak
    p6 = steepest_point_in_region(p5, rp)

    return p1, p2, p3, p4, p5, p6

class Point(object):
    def __init__(self, index, time_index, values, d_values, d2_values):
        assert index >= 0 and index < len(values)
        assert int(index) == index, "index must be an integer"
        self.i = int(index)
        self.t = float(time_index[index])
        self.v = float(values[index])
        self.dv = float(d_values[index])
        self.d2v = float(d2_values[index])

# ============================================================================ #

# utility functions for extracting features from a waveform signal
def symmetric_logarithm(x):
    return float(np.sign(x) * np.log(np.abs(x)))

def inter_quartile_range(data):
    """
    Calculate the interquartile range of data.

    Parameters
    ----------
    data : array_like
        The data to be analyzed.

    Returns
    -------
    float
        The interquartile range of data.
    """
    return np.percentile(data, 75) - np.percentile(data, 25)

def skew(data):
    """
    Calculate the skewness of data.

    Parameters
    ----------
    data : array_like
        Data to be analyzed.

    Returns
    -------
    float
        The skewness (Fisher Asymmetry) of data.
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return np.sum((data - mean)**3) / (n * std**3)

def kurtosis(data):
    """
    Calculate the kurtosis of data.

    Parameters
    ----------
    data : array_like
        The data to be analyzed.

    Returns
    -------
    float
        The kurtosis of data.
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    return np.sum((data - mean)**4) / (n * std**4)

def area_under_curve(waveform, delta):
    return np.trapz(waveform, dx=delta)

def filter_indexes(extrema_indexes, start, end):
    # get the indexes of the extrema in the region
    return list(filter(lambda i: start <= i <= end, extrema_indexes))

def local_extrema(timeseries, delta):
    is_extremum = lambda index: (timeseries[index] > timeseries[index - 1] and timeseries[index] >= timeseries[index + 1]) or (timeseries[index] < timeseries[index - 1] and timeseries[index] <= timeseries[index + 1])
    return [0] + list(filter(is_extremum, range(1, len(timeseries) - 1))) + [len(timeseries) - 1]

def zero_crossings(timeseries, delta):
    """
    Find the indexes of the points at or near zero crossings in a waveform.

    This function is not 100% precise. It finds the indexes of the points
    at or near zero crossings in a waveform. It does not find the exact
    zero crossing points.

    We generally only care about the max and min values at zero crossings,
    so this function serves its purpose in this context.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    delta : float
        The time between samples in the waveform.

    Returns
    -------
    array_like
        The indexes of the zeroes in the waveform.
    """
    return list(np.where(derivative(np.sign(timeseries), 1))[0])

# functions for getting waveform domains.

def derivative2(waveform, delta):
    """
    Calculate the second derivative of a waveform.
    """
    return derivative(derivative(waveform, delta), delta)

def derivative(waveform, delta):
    """
    Calculate the derivative of a waveform.
    """
    differential = lambda i: _differential(i, waveform, delta)
    return list(map(differential, range(len(waveform))))

def _differential(i, waveform, delta):
    if i == 0:
        return (waveform[i+1] - waveform[i]) / delta
    elif i == len(waveform)-1:
        return (waveform[i] - waveform[i-1]) / delta
    else:
        try:
            return float((waveform[i+1] - waveform[i-1]) / (2 * delta))
        except IndexError:
            raise IndexError(f"Index {i} out of range (0,{len(waveform)})")

def time_index(waveform, delta):
    """
    Return the time index for a waveform.
    """
    return list(map(lambda i: float(i * delta), range(len(waveform))))
    return duration / len(waveform)