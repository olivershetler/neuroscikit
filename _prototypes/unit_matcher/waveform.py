"""
Functions that take a single waveform as an input and return a feature.
"""

import numpy as np

# functions for extracting features from a single waveform

#TODO implement
def waveform_features(waveform, d_waveform, d2_waveform):
    """
    Extracts the following features from a waveform:
    - peak_spike_amplitude
    - spike_width
    -
    """
    pass

#TODO test
def peak_spike_amplitude(waveform):
    """
    Find the peak spike amplitude in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.

    Returns
    -------
    float
        The peak spike amplitude in the waveform.
    """
    return waveform[principal_peak_index(waveform, d_waveform)]

#TODO validate, test
def spike_width(principal_peak_index, refractory_trough_index, time_index):
    """
    Find the spike width in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.

    Returns
    -------
    float
        The spike width in the waveform.
    """
    return time_index[refractory_trough_index] - time_index[principal_peak_index]

#TODO implement


# functions for finding the key morpohlogical points in a waveform

class Point(object):
    def __init__(self, index, time_index, values, d_values, d2_values):
        self.i = index
        self.t = time_index[index]
        self.v = values[index]
        self.dv = d_values[index]
        self.d2v = d2_values[index]

#TODO implement
def morphological_point_indexes(waveform, d_waveform, d2_waveform):
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
    dict
        A dictionary of the key morphological points in the waveform of the form {point_name: index,...}
    """
    points = {}
    points["principal_peak"] = principal_peak_index(waveform, d_waveform)
    points["refractory_trough"] = refractory_trough_index(waveform, d_waveform)
    points["pre_spike_trough"] = pre_spike_trough_index(waveform, d_waveform)
    points["principal_point_of_steepest_ascent"] = principal_point_of_steepest_ascent(d_waveform, d2_waveform)
    points["principal_point_of_steepest_descent"] = principal_point_of_steepest_descent(d_waveform, d2_waveform)
    return points
#TODO implement
def principal_peak_index(waveform, d_waveform):
    """
    Find the principal peak in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : float
        The first derivative of the waveform to be analyzed.

    Returns
    -------
    int
        The index of the principal peak in the waveform.
    """
    pass
#TODO implement
def refractory_trough_index(waveform, d_waveform):
    """
    Find the refractory trough in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : float
        The first derivative of the waveform to be analyzed.

    Returns
    -------
    int
        The index of the refractory trough in the waveform.
    """
    pass
#TODO implement
def pre_spike_trough_index(waveform, d_waveform):
    """
    Find the pre-spike trough in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    d_waveform : float
        The first derivative of the waveform to be analyzed.

    Returns
    -------
    int
        The index of the pre-spike trough in the waveform.
    """
    pass
#TODO implement
def principal_point_of_steepest_ascent(d_waveform, d2_waveform):
    """
    Find the principal point of steepest ascent in a waveform.

    Parameters
    ----------
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.
    d2_waveform : array_like
        The second derivative of the waveform to be analyzed.

    Returns
    -------
    int
        The index of the principal point of steepest ascent in the waveform.
    """
    pass
#TODO implement
def principal_point_of_steepest_descent(d_waveform, d2_waveform):
    """
    Find the principal point of steepest descent in a waveform.

    Parameters
    ----------
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.
    d2_waveform : array_like
        The second derivative of the waveform to be analyzed.

    Returns
    -------
    int
        The index of the principal point of steepest descent in the waveform.
    """
    pass
#TODO implement
def refractory_point_of_steepest_descent(d_waveform, d2_waveform):
    """
    Find the refractory point of steepest descent in a waveform.

    Parameters
    ----------
    d_waveform : array_like
        The first derivative of the waveform to be analyzed.
    d2_waveform : array_like
        The second derivative of the waveform to be analyzed.

    Returns
    -------
    Point
        The point of the refractory point of steepest descent in the waveform.
    """
    pass


# utility functions for extracting features from a waveform signal

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

def find_zeroes(timeseries, delta):
    """
    Find the indexes of the zeroes in a waveform.

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
    return np.where(np.diff(np.sign(timeseries)))[0]

#TODO implement
def find_extrema(timeseries, delta):
    """
    Find the indexes of the extrema in a waveform.

    Parameters
    ----------
    waveform : array_like
        The waveform to be analyzed.
    delta : float
        The time between samples in the waveform.

    Returns
    -------
    array_like
        The indexes of the extrema in the waveform.
    """
    pass

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
            return (waveform[i+1] - waveform[i-1]) / (2 * delta)
        except IndexError:
            raise IndexError(f"Index {i} out of range (0,{len(waveform)})")

def time_index(waveform, delta):
    """
    Return the time index for a waveform.
    """
    return list(map(lambda i: i * delta, range(len(waveform))))
    return duration / len(waveform)