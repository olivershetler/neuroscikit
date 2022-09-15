import numpy as np

def waveform_mean(waveforms):
    """
    Aggregates neuron waveforms for a single recording session.

    Parameters
    ----------
    waveforms : A nested iterable containing n-dimensional waveforms. The outer iterable ranges over spikes waveforms and the inner iterable ranges over spike waveform channels (dimensions). e.g. waveforms[0][0] is the first spike waveform on the first channel. waveforms[0][1] is the first spike waveform on the second channel. etc.

    Returns
    -------
    mean_waveforms : A list of mean waveforms for each session. Each mean waveform is a numpy array of shape (n_channels, n_samples).
    """
    np_mean = lambda x: np.mean(x.squeeze(), axis=0)

    return aggregated_waveform_map(waveforms, np_mean)

def waveform_quantile_band(waveforms, quantiles=(0.25, 0.75)):
    """
    Returns a quantile band for each cell in each session.

    Parameters
    ----------
    waveforms : A nested iterable containing n-dimensional waveforms. The outer iterable ranges over spikes waveforms and the inner iterable ranges over spike waveform channels (dimensions). e.g. waveforms[0][0] is the first spike waveform on the first channel. waveforms[0][1] is the first spike waveform on the second channel. etc.

    quantiles : A tuple of quantiles to use for the upper and lower bounds of the quantile band. e.g. (0.25, 0.75) for the 25th and 75th percentiles.

    Returns
    -------
    lower_quantile_waveforms : A list of lower quantile waveforms for each session. Each lower quantile waveform is a numpy array of shape (n_channels, n_samples).

    upper_quantile_waveforms : A list of upper quantile waveforms for each session. Each upper quantile waveform is a numpy array of shape (n_channels, n_samples).
    """

    quantile = lambda x, quantile: np.quantile(x.squeeze(), quantile, axis=0)

    lower_quantile = lambda x: quantile(x, quantiles[0])
    upper_quantile = lambda x: quantile(x, quantiles[1])

    lower_quantile_waveforms = aggregated_waveform_map(waveforms, lower_quantile)
    upper_quantile_waveforms = aggregated_waveform_map(waveforms, upper_quantile)

    return lower_quantile_waveforms, upper_quantile_waveforms

def aggregated_waveform_map(waveforms, agg_map):
    """
    Aggregates cell waveforms over a single session, given an aggregation function.

    Parameters
    ----------
    waveforms : A nested iterable containing n-dimensional waveforms. The outer iterable ranges over spikes waveforms and the inner iterable ranges over spike waveform channels (dimensions). e.g. waveforms[0][0] is the first spike waveform on the first channel. waveforms[0][1] is the first spike waveform on the second channel. etc.

    agg_map : A function that takes a numpy array and returns a single value. e.g. np.mean, np.median, np.quantile, etc. over the first axis.

    Returns
    -------
    aggregated_waveforms : A list of aggregated waveforms for each session. Each aggregated waveform is a numpy array of shape (n_channels, n_samples).
    """
    aggregated_waveforms = [[] for i in range(len(waveforms))]
    # for each session
    for i in range(len(aggregated_waveforms)):
        aggregated_waveforms[i] = map(agg_map, waveforms[i])

    return aggregated_waveforms

