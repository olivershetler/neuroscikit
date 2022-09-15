import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.maps import rate_map, spike_pos, autocorrelation, filter_pos_by_speed, firing_rate_vs_time, map_blobs, occupancy_map, spatial_tuning_curve, spike_map, binary_map
from core.core_utils import make_seconds_index_from_rate


def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_arena(count=100):
    return np.random.sample(count), np.random.sample(count)

def make_velocity(count=100):
    return np.random.sample(count)


def test_rate_map():  

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

    assert type(rate_map_smooth) == np.ndarray 
    assert type(rate_map_raw) == np.ndarray
    assert rate_map_smooth.shape == rate_map_raw.shape

def test_autocorrelation():
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

    autocorr = autocorrelation(rate_map_smooth, pos_x, pos_y, arena_size)

    assert type(autocorr) == np.ndarray

def test_filter_pos_by_speed():

    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    pos_x, pos_y = make_2D_arena(len(pos_t))
    pos_v = make_velocity(len(pos_t))

    new_pos_x, new_pos_y, new_pos_t = filter_pos_by_speed(0.1, 0.9, pos_v, pos_x, pos_y, pos_t)

    assert len(new_pos_x) == len(new_pos_y)
    assert len(new_pos_y) == len(new_pos_t)

def test_firing_rate_vs_time():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)
    spk_times = make_1D_timestamps()
    rate, firing_time = firing_rate_vs_time(np.array(spk_times), np.array(pos_t), 400)

    assert type(rate) == np.ndarray

def test_map_blobs():
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

    image, n_labels, labels, centroids, field_sizes = map_blobs(rate_map_smooth)

    assert len(image) == len(labels)
    assert len(np.unique(labels)) == n_labels
    assert len(centroids) == len(field_sizes)

def test_occupancy_map():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    smoothing_factor = 5
    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)
    arena_size = (1,1)

    pos_x, pos_y = make_2D_arena(len(pos_t))

    occ_map_smoothed, occ_map_raw, coverage_map = occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

    assert len(occ_map_smoothed) == len(occ_map_raw)
    assert type(coverage_map) == np.ndarray
    
def test_spatial_tuning_curve():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    smoothing = 5
    spk_times = make_1D_timestamps()

    pos_x, pos_y = make_2D_arena(len(pos_t))
    tuned_data, spike_angles, ang_occ, bin_array = spatial_tuning_curve(pos_x, pos_y, np.array(pos_t), np.array(spk_times), smoothing)

    assert type(tuned_data) == np.ndarray
    assert type(spike_angles) == np.ndarray
    assert type(ang_occ) == tuple
    assert type(bin_array) == np.ndarray

def test_spike_pos():

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

    assert len(spikex) == len(spikey)
    assert len(spikex) == len(spiket)

def test_spike_map():

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

    spike_map_smooth, spike_map_raw = spike_map(pos_x, pos_y, pos_y, arena_size, spikex, spikey, kernlen, std)

    assert len(spike_map_smooth) == len(spike_map_raw)

def test_binary_map():
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

    assert type(binmap) == np.ndarray


if __name__ == '__main__':
    test_spike_pos()
    test_spike_map()
    test_rate_map()
    test_autocorrelation()
    test_filter_pos_by_speed()
    test_firing_rate_vs_time()
    test_map_blobs()
    test_occupancy_map()
    test_spatial_tuning_curve()
    test_binary_map()

