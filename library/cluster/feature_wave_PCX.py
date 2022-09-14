import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster import wave_PCA

import numpy as np

def feature_wave_PCX(data, iPC=1, norm=True):
    """Creates the principal components for the waveforms

    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).

    Returns:
        FD:

    """

    nCh, nSpikes, nSamp = data.shape

    wavePCData = np.zeros((nSpikes, nCh))

    if norm:
        l2norms = np.sqrt(np.sum(data ** 2, axis=-1))
        l2norms = l2norms.reshape((nCh, -1, 1))

        data = np.divide(data, l2norms)

        # removing NaNs from potential division by zero
        data[np.where(np.isnan(data) == True)] = 0

    for i, w in enumerate(data):
        av = np.mean(w, axis=0)  # row wise average
        # covariance matrix
        cv = np.cov(w.T)
        sd = np.sqrt(np.diag(cv)).T  # row wise standard deviation
        pc, _, _, _ = wave_PCA(cv)

        # standardize data to zero mean and unit variance
        wstd = (w - av) / (sd)

        # project the data onto principal component axes
        wpc = np.dot(wstd, pc)

        wavePCData[:, i] = wpc[:, iPC - 1]

    return wavePCData
