import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.data_study import (
    Study,
    Animal,
    Event,
    Spike,
)

from core.core_utils import (
    make_seconds_index_from_rate
)

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_cut_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90_1.cut')
test_tetrode_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90.1')

np.random.seed(0)

def make_1D_binary_spikes(size=100):
    spike_train = np.random.randint(2, size=size)

    return list(spike_train)

def make_2D_binary_spikes(count=20, size=100):
    spike_trains = np.zeros((count, size))

    for i in range(count):
        spike_trains[i] = np.random.randint(2, size=size).tolist()

    return spike_trains.tolist()

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_timestamps(count=20, T=2, dt=0.02):
    time = np.arange(0,T,dt)
    # spk_times = np.zeros((count, len(time)))
    spk_times = []

    for i in range(count):
        spk_count = np.random.choice(len(time), size=1)
        spk_times.append(np.random.choice(time, size=spk_count, replace=False).tolist())

    return spk_times

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

def test_event_spike_class():
    spike_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)

    # T = 2
    # dt = .02
    idx = np.random.choice(len(spike_times), size=1)[0]

    # input_dict1 = {}
    # input_dict1['sample_length'] = int(T / dt)
    # input_dict1['sample_rate'] = float(T / dt)
    spike_time = spike_times[idx]
    cluster_label = int(idx + 1)

    waves = []
    for i in range(ch_count):
        key = 'ch' + str(i+1)
        waves.append(waveforms[i][idx])

    spike_object = Spike(spike_time, cluster_label, waves)

    label = spike_object.cluster
    chan, _ = spike_object.get_peak_signal()
    waveform = spike_object.get_signal(chan-1)

    assert type(label) == int
    assert type(spike_object.spike_time) == float
    assert type(chan) == int
    assert chan <= ch_count
    assert chan > 0
    assert type(waveform) == list
    assert type(waveform[0]) == float
    assert len(waveform) == samples_per_wave
    assert len(waveforms) == ch_count

def test_animal_class():
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

    all_channel_waveforms = animal.agg_waveforms
    label = animal.agg_cluster_labels
    session = animal.get_session_data(0)
    spike_objects = animal.agg_events

    curr_count = animal.session_count

    assert type(spike_objects) == list
    assert isinstance(spike_objects[0], list)
    assert isinstance(spike_objects[0][0], Event)
    assert isinstance(spike_objects[0][0], Spike)
    assert type(session) == dict
    assert len(all_channel_waveforms) == animal.session_count
    assert len(all_channel_waveforms[0]) == ch_count
    assert type(label) == list

    animal.add_session(input_dict1[0])

    assert curr_count + 1 == animal.session_count

def test_study_class():
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
    init_dict['animal_ids'] = ['id1', 'id2', 'id3']

    study = Study(init_dict)

    animal_count = 3
    for i in range(animal_count):
        study.add_animal(input_dict1)

    animal_ids = study.get_animal_ids()

    assert type(animal_ids) == list
    assert animal_ids == init_dict['animal_ids']
    assert isinstance(study.get_animal(1), Animal)
    assert type(study.animals) == list
    assert type(study.animal_ids) == list
    assert study.timebase == timebase
    assert type(study.get_pop_spike_times()) == list




if __name__ == '__main__':
    test_event_spike_class()
    test_animal_class()
    test_study_class()
    print('we good')
