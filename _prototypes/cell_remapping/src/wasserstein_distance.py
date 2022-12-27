import numpy as np
import ot
from scipy.stats import wasserstein_distance

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