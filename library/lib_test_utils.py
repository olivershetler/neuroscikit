from datetime import datetime, timedelta
from random import sample
import numpy as np
import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.batch_space import SpikeClusterBatch
from core.spikes import SpikeCluster
from library.ensemble_space import Cell
from library.study_space import Session
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters, make_seconds_index_from_rate
from core.spikes import SpikeTrain
from core.spatial import Position2D
from core.subjects import SessionMetadata
from library.spatial_spike_train import SpatialSpikeTrain2D

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

def make_cell():
    dt = .02
    T = 2
    event_times = make_1D_timestamps()
    ch_count = 4
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
    session = Session()
    session.make_class(SpikeTrain, {'event_times': event_times, 'sample_rate': 1/dt, 'duration': T})
    inp_dict = {'event_times': event_times, 'signal': waveforms, 'session_metadata': session.session_metadata}

    cell = Cell(inp_dict)

    return cell

def make_2D_arena(count=100):
    return np.random.sample(count), np.random.sample(count)

def make_velocity(count=100):
    return np.random.sample(count)

def make_spatial_spike_train():
    T = 2
    dt = .02

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    pos_dict = {'x': x, 'y': t, 't': t, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)}

    spike_dict = {}
    spike_dict['duration'] = int(T)
    spike_dict['sample_rate'] = float(1 / dt)
    spike_dict['events_binary'] = []
    spike_dict['event_times'] = event_times

    session = Session()
    session.set_smoothing_factor(3)
    session_metadata = session.session_metadata

    spike_train = session.make_class(SpikeTrain, spike_dict)
    position = session.make_class(Position2D, pos_dict)

    spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'spike_train': spike_train, 'position': position})

    spatial_spike_train.session_metadata.session_object.set_smoothing_factor(3)

    return spatial_spike_train, session_metadata

    T = 2
    dt = .02

    event_times = make_1D_timestamps(T, dt)
    t = make_seconds_index_from_rate(T, 1/dt)
    x, y = make_2D_arena(count=len(t))

    pos_dict = {'x': x, 'y': t, 't': t, 'arena_height': max(y) - min(y), 'arena_width': max(x) - min(x)}

    spike_dict = {}
    spike_dict['duration'] = int(T)
    spike_dict['sample_rate'] = float(1 / dt)
    spike_dict['events_binary'] = []
    spike_dict['event_times'] = event_times

    session = Session()
    session_metadata = session.session_metadata

    spike_train = session.make_class(SpikeTrain, spike_dict)
    position = session.make_class(Position2D, pos_dict)

    spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'spike_train': spike_train, 'position': position})

    spatial_spike_train.session_metadata.session_object.set_smoothing_factor(3)

    return spatial_spike_train, session_metadata