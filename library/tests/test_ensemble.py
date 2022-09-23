import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.ensembles import CellEnsemble, CellPopulation, SpikeTrainBatch, SpikeClusterBatch, Cell
from core.core_utils import *
from core.subjects import SessionMetadata
from core.spikes import Spike, SpikeCluster, SpikeTrain
from library.ensembles import Cell
from x_io.rw.axona.batch_read import make_session



def test_spike_cluster_batch_class():
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

    all_channel_waveforms = spike_cluster_batch.get_all_channel_waveforms()
    rate = spike_cluster_batch.get_single_cluster_firing_rate(event_labels[0])
    labels = spike_cluster_batch.get_cluster_labels()
    unique_labels = spike_cluster_batch.get_unique_cluster_labels()
    # spk_count = spike_cluster_batch.get_cluster_spike_count()
    single_channel_waveform = spike_cluster_batch.get_single_channel_waveforms(1)
    # spike_objects = spike_cluster_batch.get_spike_object_instances()
    rates = spike_cluster_batch.get_all_cluster_firing_rates()
    spike_clusters = spike_cluster_batch.get_spike_cluster_instances()
    count, cluster_event_times, cluster_waveforms = spike_cluster_batch.get_single_spike_cluster_instance(event_labels[0])
    single_cluster_spike_objects = spike_cluster_batch.get_single_spike_cluster_objects(event_labels[0])
    cluster_spike_objects = spike_cluster_batch.get_spike_cluster_objects()

    ids = np.where(np.array(event_labels) == event_labels[0])[0]
    assert np.array(cluster_event_times).all() == np.array(event_times)[ids].all()
    assert np.array(cluster_waveforms).all() == np.array(waveforms[2]).all()
    assert len(unique_labels) <= cluster_count
    assert type(rates) == list
    assert rate in rates
    assert type(single_cluster_spike_objects) == list
    assert isinstance(single_cluster_spike_objects[0], Spike)
    assert type(cluster_spike_objects) == list
    assert isinstance(cluster_spike_objects[0], list)
    assert isinstance(cluster_spike_objects[0][0], Spike)
    assert type(spike_clusters) == list
    assert isinstance(spike_clusters[0], SpikeCluster)
    assert type(single_channel_waveform) == list
    assert type(single_channel_waveform[0]) == list
    assert type(single_channel_waveform[0][0]) == float
    assert len(all_channel_waveforms) == ch_count
    assert len(single_channel_waveform) == len(event_times)
    assert type(labels) == list
    assert type(rate) == float

def test_spike_train_batch_class():
    event_times = make_2D_timestamps()

    T = 2
    dt = .02

    input_dict1 = {}
    input_dict1['duration'] = int(T)
    input_dict1['sample_rate'] = float(1 / dt)
    input_dict1['events_binary'] = []
    input_dict1['event_times'] = event_times

    spike_train1 = SpikeTrainBatch(input_dict1)
    rate1 = spike_train1.get_average_event_rate()
    rate_list1 = spike_train1.get_indiv_event_rate()
    spike_train1.get_binary()
    instances1 = spike_train1.get_spike_train_instances()

    assert type(rate1) == np.float64
    assert type(rate_list1) == list
    assert type(spike_train1.events_binary) == list
    assert type(spike_train1.event_times) == list
    assert type(spike_train1.event_labels) == list
    assert type(spike_train1.events_binary[0]) == list
    assert type(spike_train1.event_times[0]) == list
    assert isinstance(instances1[0], SpikeTrain) == True

    events_binary2 = make_2D_binary_spikes()

    input_dict2 = {}
    input_dict2['duration'] = int(T)
    input_dict2['sample_rate'] = float(1 / dt)
    input_dict2['events_binary'] = events_binary2
    input_dict2['event_times'] = []

    spike_train2 = SpikeTrainBatch(input_dict2)

    spike_train2.get_event_times()
    rate2 = spike_train2.get_average_event_rate()
    rate_list2 = spike_train2.get_indiv_event_rate()
    instances2 = spike_train2.get_spike_train_instances()

    assert type(rate2) == np.float64
    assert type(rate_list2) == list
    assert type(spike_train2.events_binary) == list
    assert type(spike_train2.event_times) == list
    assert type(spike_train2.event_labels) == list
    assert type(spike_train2.events_binary[0]) == list
    assert type(spike_train2.event_times[0]) == list
    assert isinstance(instances2[0], SpikeTrain) == True

def test_cell_ensemble():
    cells = {}
    for i in range(5):
        events = make_1D_timestamps()
        waveforms = make_waveforms
        session_metadata = SessionMetadata({'session_id': 'id0'})
        cell = Cell({'events': events, 'signal': waveforms, 'session_metadata': session_metadata})
        cells['cell_'+ str(i+1)] = cell

    ensemble = CellEnsemble(cells)

    assert isinstance(ensemble, CellEnsemble)
    assert type(ensemble.cells) == list

    events = make_1D_timestamps()
    waveforms = make_waveforms
    session_metadata = SessionMetadata({'session_id': 'id0'})
    cell_new = Cell({'events': events, 'signal': waveforms, 'session_metadata': session_metadata})

    ensemble.add_cell(cell_new)

    assert ensemble.cells[-1] == cell_new

def test_cell_population():
    cells = {}
    for i in range(5):
        events = make_1D_timestamps()
        waveforms = make_waveforms
        session_metadata = SessionMetadata({'session_id': 'id0'})
        cell = Cell({'events': events, 'signal': waveforms, 'session_metadata': session_metadata})
        cells['cell_'+ str(i+1)] = cell

    ensemble = CellEnsemble(cells)

    population = CellPopulation()

    assert type(population.ensembles) == list

