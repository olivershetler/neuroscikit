import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.data_study import Study, Animal
from core.core_utils import make_seconds_index_from_rate
from _prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times


def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_waveforms(channel_count, spike_count, samples_per_wave):
    waveforms = np.zeros((channel_count, spike_count, samples_per_wave))

    for i in range(channel_count):
        for j in range(samples_per_wave):
            waveforms[i,:,j] = np.random.randint(-20,20,size=spike_count).tolist()
            for k in range(10):
                waveforms[i,:,j] += np.random.rand() 

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
    ch_count = 4
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
    cluster_labels = make_clusters(spike_times, cluster_count)
    waves.append(waveforms)
    input_dict1[1]['spike_times'] = spike_times
    input_dict1[1]['cluster_labels'] = cluster_labels

    for j in range(0,2):
        for i in range(ch_count):
            key = 'ch' + str(i+1)
            input_dict1[j][key] = waves[j][i]

    init_dict = {}
    init_dict['sample_length'] = T
    init_dict['sample_rate'] = float(1 / dt)
    init_dict['animal_ids'] = ['id1', 'id2']

    study = Study(init_dict)

    animal_count = 2
    for i in range(animal_count):
        study.add_animal(input_dict1)

    animal = study.animals[0]

    good_cells, good_sorted_waveforms = sort_cell_spike_times(animal.agg_events, animal.agg_cluster_labels, animal.agg_waveforms)

    assert type(good_cells) == list
    assert type(good_sorted_waveforms) == list
    assert len(good_cells) == len(good_sorted_waveforms)

if __name__ == '__main__':
    test_sort_cell_spike_times()