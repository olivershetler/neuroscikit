def compare_unit_means(unit1, unit2):
    """Return the mean squared difference between the mean waveforms of two units.

    Parameters
    ----------
    unit1 : int
        The first unit ID.
    unit2 : int
        The second unit ID.

    Returns
    -------
    float
        The mean squared difference between the mean waveforms of the two units.
    """
    return mean_squared_difference(unit_mean(unit1), unit_mean(unit2))


def mean_squared_difference(a, b):
    """Return the mean squared difference between two waveforms.

    Parameters
    ----------
    a : np.ndarray
        The first waveform.
    b : np.ndarray
        The second waveform.

    Returns
    -------
    float
        The mean squared difference between the two waveforms.
    """
    return np.mean((a - b) ** 2)


def unit_mean(unit, **kwargs):
    """Return the mean waveform of a single unit.

    Parameters
    ----------
    unit:

    Returns
    -------
    np.ndarray
        The mean waveform of the unit.
    """
    n_spikes, spike_times, waveforms = unit.get_single_spike_cluster_instance(unit)
    return np.array(waveforms).mean(axis=1)