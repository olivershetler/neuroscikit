import numpy as np

"""
This module contains functions for extracting waveform features for each unit.

- Functions for extracting a long feature vector for each unit.
- Functions for summarizing aggregated unit features.
"""

import numpy as np

from _prototypes.unit_matcher.spike import (
    spike_features
)
from core.spikes import SpikeCluster

# Utility functions for comparing distributions

#TODO test
def jensen_shannon_distance(P:np.array, Q:np.array):
    """Compute the Jensen-Shannon distance between two probability distributions.

    Input
    -----
    P, Q : 2D arrays (sample_size, dimensions)
        Probability distributions of equal length that sum to 1
    """
    P_sample_size, P_dimensions = P.shape
    Q_sample_size, Q_dimensions = Q.shape
    assert P_dimensions == Q_dimensions, f"Dimensionality of P ({P_dimensions}) and Q ({Q_dimensions}) must be equal"
    dimensions = P_dimensions

    M = compute_mixture(P, Q)

    if dimensions == 1:
        _kldiv = lambda A, B: np.sum([v for v in A * np.log(A/B) if not np.isnan(v)])
    if dimensions > 1:
        _kldiv = lambda A, B: multivariate_kullback_leibler_divergence(A, B)
    else:
        raise ValueError(f"Dimensionality of P ({P_dimensions}) and Q ({Q_dimensions}) must be greater than 0")

    kl_pm = _kldiv(P, M)
    print(f"kl_pm: {kl_pm}")
    kl_qm = _kldiv(Q, M)
    print(f"kl_qm: {kl_qm}")

    jensen_shannen_divergence = (kl_pm + kl_qm)/2
    print("JSD", jensen_shannen_divergence)

    return np.sqrt(jensen_shannen_divergence)

def compute_mixture(P:np.array, Q:np.array):
    """Compute the mixture distribution between two probability distributions.

    Input
    -----
    P, Q : 2D arrays (sample_size, dimensions)
        Probability distributions of equal length that sum to 1
    """
    P_sample_size, P_dimensions = P.shape
    Q_sample_size, Q_dimensions = Q.shape

    from random import randint, sample

    #M = np.array([_mixture_sample(P, Q) for _ in range(max(P_sample_size, Q_sample_size))])

    ss = int(min(P_sample_size, Q_sample_size)/2)

    M = np.concatenate((P[0:ss,:], Q[ss:,:]), axis=0)

    print("M", M.shape)

    return M

def _mixture_sample(P:np.array, Q:np.array):
    """Sample a mixture distribution between two probability distributions.

    Input
    -----
    P, Q : 2D arrays (sample_size, dimensions)
        Probability distributions of equal length that sum to 1
    """
    P_sample_size, P_dimensions = P.shape
    Q_sample_size, Q_dimensions = Q.shape

    from random import randint
    choice = randint(0, 1)
    if choice == 0:
        return P[randint(0, P_sample_size-1),:]
    else:
        return Q[randint(0, Q_sample_size-1),:]

def kullback_leibler_divergence(P, Q):
    return np.sum(list(filter(lambda x: not np.isnan(x), P * np.log(P/Q))))

def multivariate_kullback_leibler_divergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (sample_size, dimensionality)
        Samples from distribution P, which typically represents the true
        distribution.
    y : 2D array (sample_size, dimensionality)
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    Adapted from https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    """
    #from scipy.spatial import KDTree
    #from sklearn.neighbors import KDTree
    from sklearn.neighbors import BallTree
    from scipy.spatial.distance import cosine, correlation
    from sklearn.metrics.pairwise import cosine_distances

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    x_sample_size, x_dimensions = x.shape
    y_sample_size, y_dimensions = y.shape

    assert(x_dimensions == y_dimensions), f"Both samples must have the same number of dimensions. x has {x_dimensions} dimensions, while y has {y_dimensions} dimensions."

    d = x_dimensions
    n = x_sample_size
    m = y_sample_size

    # Build a KD tree representation of the samples and find the nearest
    # neighbour of each point in x.
    xtree = BallTree(x, metric='l1')
    ytree = BallTree(y, metric='l1')

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    R = xtree.query(x, k=3)[0][:,0:]
    r = np.array([min(list(filter(lambda x: x!=0, R))) for R in R])
    S = ytree.query(x, k=3)[0][:,0:]
    #for i in range(n):
    #    if s[i] == 0:
    #        s[i] = r[i]
    s = np.array([min(list(filter(lambda x: x!=0, s))) for s in S])
    kl_div = sum(np.log2(s/r)) * d / n + np.log2(m / (n - 1.))
    return max(kl_div, 0)

def spike_level_feature_array(unit: SpikeCluster, time_step):
    """Compute features for each spike in a unit.

    This function mutates a unit (SpikeCluster) by adding a spike-level feature dictionary to the unit.features attribute.

    Input
    -----
    unit : 2D array (spike_size, dimensions)
        Waveforms for each spike in a unit.
    time_step : float
        Time between samples in the waveform.

    Output
    ------
    features : dict
        Dictionary of features for each spike in the unit.
    """
    spikes = unit.get_spike_object_instances()
    #features = {}
    feature_array = []
    for spike in spikes:
        #spike.features = spike_features(spike, time_step)
        feature_vector = list(spike_features(spike, time_step).values())
        feature_array.append(feature_vector)
    return np.array(feature_array)


"""
['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'wminkowski', 'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc']
"""