import numpy as np
import ot
import os, sys
from scipy.stats import wasserstein_distance

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.backend import get_backend, NumpyBackend
from _prototypes.cell_remapping.src.utils import list_to_array


# NEEDS UPDATING
def compute_wasserstein_distance(X, Y):
    # distance = wasserstein_distance(prev, ses)
    coords = np.array([X.flatten(), Y.flatten()]).T
    coordsSqr = np.sum(coords**2, 1)
    M = coordsSqr[:, None] + coordsSqr[None, :] - 2*coords.dot(coords.T)
    M[M < 0] = 0
    M = np.sqrt(M)
    radius = 0.2
    I = 1e-5 + np.array((X)**2 + (Y)**2 < radius**2, dtype=float)
    I /= np.sum(I)

    l2dist = np.sqrt(np.sum((I)**2))
    wass = ot.sinkhorn2(I.flatten(), I.flatten(), M, 1.0)

    # assert X.shape == Y.shape
    # n = X.shape[0]
    # d = cdist(X, Y)
    # assignment = linear_sum_assignment(d)
    # wass = d[assignment].sum() / n
    # return wass

    return wass, l2dist, I

def single_point_wasserstein(object_coords, rate_map_obj):
    # gets arena height and width as inches or whatever unit they were entered in the position file
    arena_height, arena_width = rate_map_obj.arena_size
    arena_height = arena_height[0]
    arena_width = arena_width[0]

    rate_map, _ = rate_map_obj.get_rate_map()

    # gets rate map dimensions (64, 64)
    y, x = rate_map.shape

    # normalize rate map
    total_mass = np.sum(rate_map)
    if total_mass != 1:
        rate_map = rate_map / total_mass

    # this is the step size between each bucket, so 0 to height step is first bucket, height_step to height_step*2 is next and so one
    height_step = arena_height/x
    width_step = arena_width/y

    # convert height/width to arrayswith 64 bins, this gets us our buckets
    height = np.arange(0,arena_height, height_step)
    width = np.arange(0,arena_width, width_step)

    # because they are buckets, i figured I will use the midpoint of the pocket when computing euclidean distances
    height_bucket_midpoints = height + height_step/2
    width_bucket_midpoints = width + width_step/2

    # these are the coordinates of the object on a 64,64 array so e.g. (0,32)
    obj_x = object_coords['x']
    obj_y = object_coords['y']

    # loop through each bucket and compute the euclidean distance between the object and the bucket
    # then multiply that distance by the rate map value at that bucket
    weighted_dists = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            pt = (width_bucket_midpoints[i], height_bucket_midpoints[j])
            dist = np.linalg.norm(np.array((obj_y, obj_x)) - np.array(pt))
            weighted_dists[i,j] = dist * rate_map[i,j]

    # then sum
    return np.sum(weighted_dists)


# https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
def sliced_wasserstein(X, Y, num_proj):
    dim = X.shape[1]
    estimates = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein
        estimates.append(wasserstein_distance(X_proj, Y_proj))
    return np.mean(estimates)

########################################################################################################################################################
""" Code below from POT """
########################################################################################################################################################

def wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        nx = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = nx.full(u_values.shape, 1. / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = nx.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = nx.full(v_values.shape, 1. / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = nx.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = nx.argsort(u_values, 0)
        u_values = nx.take_along_axis(u_values, u_sorter, 0)

        v_sorter = nx.argsort(v_values, 0)
        v_values = nx.take_along_axis(v_values, v_sorter, 0)

        u_weights = nx.take_along_axis(u_weights, u_sorter, 0)
        v_weights = nx.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = nx.cumsum(u_weights, 0)
    v_cumweights = nx.cumsum(v_weights, 0)

    qs = nx.sort(nx.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    qs = nx.zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = nx.abs(u_quantiles - v_quantiles)

    if p == 1:
        return nx.sum(delta * nx.abs(diff_quantiles), axis=0)
    return nx.sum(delta * nx.power(diff_quantiles, p), axis=0)


def pot_sliced_wasserstein(X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                projections=None, seed=None, log=False):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> reg = 0.1
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """

    X_s, X_t = list_to_array(X_s, X_t)

    if a is not None and b is not None and projections is None:
        nx = get_backend(X_s, X_t, a, b)
    elif a is not None and b is not None and projections is not None:
        nx = get_backend(X_s, X_t, a, b, projections)
    elif a is None and b is None and projections is not None:
        nx = get_backend(X_s, X_t, projections)
    else:
        nx = get_backend(X_s, X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = nx.full(n, 1 / n, type_as=X_s)
    if b is None:
        b = nx.full(m, 1 / m, type_as=X_s)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, backend=nx, type_as=X_s)

    X_s_projections = nx.dot(X_s, projections)
    X_t_projections = nx.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (nx.sum(projected_emd) / n_projections) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res

def get_random_projections(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to ue for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == 'numpy':
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections

def quantile_function(qs, cws, xs):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    nx = get_backend(qs, cws)
    n = xs.shape[0]
    if nx.__name__ == 'torch':
        # this is to ensure the best performance for torch searchsorted
        # and avoid a warninng related to non-contiguous arrays
        cws = cws.T.contiguous()
        qs = qs.T.contiguous()
    else:
        cws = cws.T
        qs = qs.T
    idx = nx.searchsorted(cws, qs).T
    return nx.take_along_axis(xs, nx.clip(idx, 0, n - 1), axis=0)