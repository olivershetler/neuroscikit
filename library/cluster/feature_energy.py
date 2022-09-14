import numpy as np 

def feature_energy(data):
    """
    Normlized energy calculation discussed from:
    Quantitative measures of cluster quality for use in extracellular recordings by Redish et al.

    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).

    Returns:
        E:

    """
    # energy sqrt of the sum of the squares of each point of the waveform, divided by number of samples in waveform
    # energy and first principal component coefficient
    nSamp = data.shape[-1]

    sum_squared = np.sum(data ** 2, axis=-1)

    if len(np.where(sum_squared < 0)[0]) > 0:
        raise ValueError('summing squared values produced negative number!')

    E = np.divide(np.sqrt(sum_squared), np.sqrt(nSamp))  # shape: channel numbers x spike number
    # E[np.where(E == 0)] = 1  # remove any 0's so we avoid dividing by zero

    return E.T
