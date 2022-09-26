import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.cluster.features import _wave_PCA, feature_energy, feature_wave_PCX, create_features
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters
from core.spikes import SpikeCluster
from library.batch_space import SpikeClusterBatch

def make_spike_cluster_batch():
    event_times = make_1D_timestamps()
    ch_count = 4
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
    cluster_count = 10

    event_labels = make_clusters(event_times, cluster_count)

    T = 2
    dt = .02

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['event_times'] = event_times
    input_dict1['event_labels'] = event_labels


    for i in range(ch_count):
        key = 'channel_' + str(i+1)
        input_dict1[key] = waveforms[i]

    spike_cluster_batch = SpikeClusterBatch(input_dict1)

    return spike_cluster_batch

def make_spike_cluster():
    event_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)

    T = 2
    dt = .02
    idx = np.random.choice(len(event_times), size=1)[0]

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['event_times'] = event_times
    input_dict1['cluster_label'] = int(idx + 1)


    for i in range(ch_count):
        key = 'channel_' + str(i+1)
        input_dict1[key] = waveforms[i]

    spike_cluster = SpikeCluster(input_dict1)

    return spike_cluster


def test__wave_PCA():
    cv = np.eye(5)

    pc, rpc, ev, rev = _wave_PCA(cv)

    assert type(pc) == np.ndarray
    assert type(rpc) == np.ndarray
    assert type(ev) == np.ndarray
    assert type(rev) == np.ndarray

def test_feature_wave_PCX():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()
    
    wavePCData = feature_wave_PCX(spike_cluster)
    wavePCData_batch = feature_wave_PCX(spike_cluster_batch)

    assert type(wavePCData) == np.ndarray
    assert type(wavePCData_batch) == np.ndarray

def test_create_features():
    spike_cluster = make_spike_cluster_batch()
    FD = create_features(spike_cluster)
    assert type(FD) == np.ndarray

    spike_cluster_batch = make_spike_cluster_batch()
    FD = create_features(spike_cluster_batch)
    assert type(FD) == np.ndarray

def test_feature_energy():
    spike_cluster = make_spike_cluster()
    spike_cluster_batch = make_spike_cluster_batch()

    E = feature_energy(spike_cluster)
    assert type(E) == np.ndarray

    E = feature_energy(spike_cluster_batch)
    assert type(E) == np.ndarray

if __name__ == '__main__':
    test_feature_wave_PCX()
    test_feature_energy()
    test__wave_PCA()
    test_create_features()




