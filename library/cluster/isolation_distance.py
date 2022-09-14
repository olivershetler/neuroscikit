import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster import mahal

import numpy as np

def isolation_distance(FD, ClusterSpikes):
    """Measures the Isolation Distance, a cluster quality metric.

    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        IsoDist: the isolation distance

    """

    nSpikes = FD.shape[0]

    nClusterSpikes = len(ClusterSpikes)

    if nClusterSpikes > nSpikes / 2:
        IsoDist = np.NaN  # not enough out-of-cluster-spikes - IsoD undefined

    else:

        # InClu = ClusterSpikes
        OutClu = np.setdiff1d(np.arange(nSpikes), ClusterSpikes)

        # compute mahalanobis distances
        m = mahal(FD, FD[ClusterSpikes, :])

        mNoise = m[OutClu]  # mahal dist of all other spikes

        # calculate point where mD of other spikes = n of this cell
        sorted_values = np.sort(mNoise)
        IsoDist = sorted_values[nClusterSpikes - 1]

    return IsoDist
