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