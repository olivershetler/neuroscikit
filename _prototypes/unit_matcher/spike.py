"""
Functions for extracting features for a single spike (n-dimensional waveforms).

- A function for extracting all the features in the waveform module.
- A function for estimating source locations for a spike from waveform features (amplitude).
"""

from .waveform import waveform_features

#TODO implement
def extract_spike_features():
    pass

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

#TODO test
def waveform_level_features(spike:dict):
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
    wf = lambda item: (item[0], waveform_features(item[1]))
    return dict(map(wf, spike.items()))


