import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate, make_1D_timestamps
from library.lib_utils import make_2D_arena, make_spatial_spike_train
from library.maps import binary_map, rate_map, occupancy_map, spatial_tuning_curve
from library.scores import border_score, grid_score, hd_score
from library.lib_utils import make_2D_arena

def test_border_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    bscore = border_score(spatial_spike_train)

    assert type(bscore) == tuple

def test_grid_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    gscore = grid_score(spatial_spike_train)

    assert type(gscore) == np.float64

def test_hd_score():
    spatial_spike_train, session_metadata = make_spatial_spike_train()

    smooth_hd = hd_score(spatial_spike_train)

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


