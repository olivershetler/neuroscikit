import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster import mahal

import numpy as np
from scipy.stats import chi2

def L_ratio(FD, ClusterSpikes):
    """Measures the L-Ratio, a cluster quality metric.

    Args:
        FD (ndarray): N by D array of feature vectors (N spikes, D dimensional feature space)
        ClusterSpikes (ndarray): Index into FD which lists spikes from the cell whose quality is to be evaluated.

    Returns:
        L :
        Lratio :
        df: degrees of freedom (number of features)

    """

    nSpikes = FD.shape[0]

    nClusterSpikes = len(ClusterSpikes)

    # mark spikes which are not a part of the cluster as Noise
    NoiseSpikes = np.setdiff1d(np.arange(nSpikes), ClusterSpikes)

    # compute mahalanobis distances
    m = mahal(FD, FD[ClusterSpikes, :])

    df = FD.shape[1]

    L = np.sum(1 - chi2.cdf(m[NoiseSpikes], df))
    Lratio = L / nClusterSpikes

    return L, Lratio, df
