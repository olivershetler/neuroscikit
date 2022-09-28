import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from library.maps import binary_map, rate_map, spike_pos, occupancy_map, spatial_tuning_curve
from library.scores import border_score, grid_score, hd_score, shuffle_spikes
from library.lib_utils import make_2D_arena

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_arena(count=100):
    return np.random.sample(count), np.random.sample(count)

def test_border_score():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    smoothing_factor = 5
    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)
    arena_size = (1,1)

    spk_times = make_1D_timestamps()
    pos_x, pos_y = make_2D_arena(len(pos_t))

    spikex, spikey, spiket, _ = spike_pos(spk_times, pos_x, pos_y, pos_t, pos_t, False, False)

    rate_map_smooth, rate_map_raw = rate_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)    

    binmap = binary_map(rate_map_smooth)

    bscore = border_score(binmap, rate_map_smooth)

    assert type(bscore) == tuple

def test_grid_score():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    smoothing_factor = 5
    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)
    arena_size = (1,1)

    spk_times = make_1D_timestamps()
    pos_x, pos_y = make_2D_arena(len(pos_t))

    spikex, spikey, spiket, _ = spike_pos(spk_times, pos_x, pos_y, pos_t, pos_t, False, False)

    rate_map_smooth, rate_map_raw = rate_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)    

    occ_map_smoothed, occ_map_raw, coverage_map = occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

    gscore = grid_score(occ_map_smoothed, spk_times, pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)

    assert type(gscore) == np.float64

def test_hd_score():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    smoothing = 5
    spk_times = make_1D_timestamps()

    pos_x, pos_y = make_2D_arena(len(pos_t))
    tuned_data, spike_angles, angular_occupancy, bin_array = spatial_tuning_curve(pos_x, pos_y, np.array(pos_t), np.array(spk_times), smoothing)
    smooth_hd =  hd_score(spike_angles, window_size=23)

    assert type(smooth_hd) == np.ndarray

# def test_shuffle_spikes():
#     T = 10
#     dt = .01
#     pos_t = make_seconds_index_from_rate(T, 1/dt)

#     spk_times = make_1D_timestamps(T=T, dt=dt)

#     pos_x, pos_y = make_2D_arena(len(pos_t))
#     shuffled_spikes = shuffle_spikes(np.array(spk_times), np.array(pos_x), np.array(pos_y), pos_t)

#     assert len(shuffled_spikes) == len(spk_times)
#     assert type(shuffled_spikes) == np.ndarray

if __name__ == '__main__':
    test_border_score()
    test_grid_score()
    test_hd_score()
    # test_shuffle_spikes()


