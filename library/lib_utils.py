from datetime import datetime, timedelta
from random import sample
import numpy as np
import os, sys
from core.subjects import SessionMetadata

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.batch_space import SpikeClusterBatch
from core.spikes import SpikeCluster
from library.ensemble_space import Cell
from library.study_space import Session
from core.core_utils import make_1D_timestamps, make_waveforms, make_clusters
from core.spikes import SpikeTrain

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
