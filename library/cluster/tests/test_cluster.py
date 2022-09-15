import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from core.data_study import Animal
from library.cluster import feature_wave_PCX, wave_PCA, create_features, feature_energy, isolation_distance, L_ratio, mahal

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_arena(count=100):
    return np.random.sample(count), np.random.sample(count)

def make_waveforms(channel_count, spike_count, samples_per_wave):
    waveforms = np.zeros((channel_count, spike_count, samples_per_wave))

    for i in range(channel_count):
        for j in range(samples_per_wave):
            waveforms[i,:,j] = np.random.randint(-20,20,size=spike_count).tolist()

    return waveforms.tolist()

def make_clusters(timestamps, cluster_count):
    cluster_labels = []
    for i in range(len(timestamps)):
        idx = np.random.choice(cluster_count, size=1)[0]
        cluster_labels.append(int(idx))
    return cluster_labels

def test_wave_PCA():
    cv = np.eye(5)

    pc, rpc, ev, rev = wave_PCA(cv)

    assert type(pc) == np.ndarray
    assert type(rpc) == np.ndarray
    assert type(ev) == np.ndarray
    assert type(rev) == np.ndarray

def test_feature_wave_PCX():
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
 
    wavePCData = feature_wave_PCX(np.array(waveforms))

    assert type(wavePCData) == np.ndarray

def test_create_features():
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    FD = create_features(np.array(waveforms))

    assert type(FD) == np.ndarray

def test_feature_energy():
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    E = feature_energy(np.array(waveforms))

    assert type(E) == np.ndarray

def test_isolation_distance():
    T = 2
    dt = .02
    timebase = make_seconds_index_from_rate(T, 1/dt)
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    FD = create_features(np.array(waveforms))

    cluster_labels = make_clusters(timebase, 10)

    iso_dist = isolation_distance(FD, cluster_labels)

    assert type(iso_dist) == float

def test_L_ratio():
    T = 2
    dt = .02
    timebase = make_seconds_index_from_rate(T, 1/dt)
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    FD = create_features(np.array(waveforms))

    cluster_labels = make_clusters(timebase, 10)
    L, ratio, df = L_ratio(FD, cluster_labels)

    assert type(L) == np.float64
    assert type(ratio) == np.float64 
    assert type(df) == int 

def test_mahal():
    T = 2
    dt = .02
    timebase = make_seconds_index_from_rate(T, 1/dt)
    ch_count = 8
    samples_per_wave = 50

    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    FD = create_features(np.array(waveforms))

    cluster_labels = make_clusters(timebase, 10)
    d = mahal(FD, FD[cluster_labels, :])

    assert type(d) == np.ndarray

if __name__ == '__main__':
    test_feature_wave_PCX()
    test_feature_energy()
    test_wave_PCA()
    test_create_features()
    test_mahal()
    test_isolation_distance()
    test_L_ratio()




