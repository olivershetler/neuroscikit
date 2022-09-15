from audioop import avg
import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from core.data_study import Animal
from library.spike import sort_cell_spike_times, find_burst, avg_spike_burst

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


def test_sort_cell_spike_times():
    waves = []
    spike_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    waves.append(waveforms)
    cluster_count = 3
    cluster_labels = make_clusters(spike_times, cluster_count)

    T = 2
    dt = .02
    timebase = make_seconds_index_from_rate(T, 1/dt)
    idx = np.random.choice(len(spike_times), size=1)[0]

    input_dict1 = {}
    input_dict1['timebase'] = timebase
    input_dict1['id'] = 'id'
    input_dict1[0] = {}
    input_dict1[1] = {}
    input_dict1[0]['spike_times'] = spike_times
    input_dict1[0]['cluster_labels'] = cluster_labels


    spike_times = make_1D_timestamps()
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)
    waves.append(waveforms)
    cluster_labels = make_clusters(spike_times, cluster_count)
    input_dict1[1]['spike_times'] = spike_times
    input_dict1[1]['cluster_labels'] = cluster_labels

    for j in range(0,2):
        for i in range(ch_count):
            key = 'ch' + str(i+1)
            input_dict1[j][key] = waves[j][i]

    animal = Animal(input_dict1)
    sort_cell_spike_times(animal)

    assert animal.agg_sorted_events != None
    assert type(animal.agg_sorted_events) == list
    assert animal.agg_sorted_waveforms != None
    assert type(animal.agg_sorted_waveforms) == list

def test_find_burst():
    ts = make_1D_timestamps()
    bursts, single_spikes = find_burst(ts)

    assert type(bursts) == np.ndarray 
    assert type(single_spikes) == np.ndarray

def test_avg_spike_burst():
    ts = make_1D_timestamps()
    bursts, single_spikes = find_burst(ts)
    avg_burst = avg_spike_burst(np.array(ts), bursts, single_spikes)

    assert type(avg_burst) == float



if __name__ == '__main__':
    test_sort_cell_spike_times()
    test_find_burst()
    test_avg_spike_burst()


