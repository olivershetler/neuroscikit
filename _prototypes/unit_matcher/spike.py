"""
Functions for extracting features for a single spike (n-dimensional waveforms).

- A function for extracting all the features in the waveform module.
- A function for estimating source locations for a spike from waveform features (amplitude).
"""

from .waveform import waveform_features

def extract_spike_features(spike):
    """
    Extract all the features for a single spike.

    Parameters
    ----------
    spike : dictionary
        The spike waveform to be analyzed.

    Returns
    -------
    dict
        A dictionary of the waveform level features of the form {feature_name: value,...}
    """
    return waveform_level_features(spike)

#TODO implement
def localize_tetrode_source(peak_amplitude:dict, tetrode_positions:dict):
    """
    Estimate the source location of a spike from the tetrode positions and the
    peak amplitudes of the spike waveform.

    Parameters
    ----------
    peak_amplitude : dict
        A dictionary of the maximum peak amplitudes for each channel in the tetrode of the form {channel_(int): amplitude,...}
    tetrode_positions : dict
        A dictionary of the positions of each channel in the tetrode of the form {channel_(int): (x,y,z),...}

    Returns
    -------
    tuple
        The estimated (x,y,z) source location of the spike.
    """
    pass

def _peak_spike_amplitudes(features):
    return features["peak_spike_amplitude"]

#TODO implement
def reduce_dimensionality(features):
    """
    Reduce the dimensionality of the waveform features.

    Parameters
    ----------
    features : dict
        A dictionary of the waveform level features of the form {feature_name: value,...}

    Returns
    -------
    dict
        A dictionary of the waveform level features of the form {feature_name: value,...}
    """
    pass

# NOTE: the waveform_level_features functions currently just takes
# the features for the channel with the largest peak amplitude.
# This will eventually be updated to take the features for all channels
# and then reduce the dimensionality of the using to-be-determined criteria.
def waveform_level_features(spike:dict, delta):
    """
    Extract waveform level features from a spike waveform.

    Parameters
    ----------
    spike : dictionary
        The spike waveform to be analyzed.

    Returns
    -------
    dict
        A dictionary of the waveform level features of the form {feature_name: value,...}
    """
    #wf = lambda item: (item[0], waveform_features(item[1]))
    #return dict(map(wf, spike.items()))
    spike_peaks = dict(map(lambda item: (item[0], max(item[1])), spike.items()))
    channel_with_max_peak = max(spike_peaks, key=spike_peaks.get)
    return waveform_features(spike[channel_with_max_peak], delta)

