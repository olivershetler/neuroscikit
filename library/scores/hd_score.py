
import numpy as  np
import matplotlib.pyplot as plt
import math

def _moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def _get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = _moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = _moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

# called by batch_process module only
def hd_score(angles, window_size=23):
    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis

    # IF THIS IS SLOW TRY NP.HISTOGRAM INSTEAD OF PLT.HIST
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = _get_rolling_sum(binned_hd, window=window_size)
    plt.close()
    return smooth_hd
