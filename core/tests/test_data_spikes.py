import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.data_spikes import (
    SpikeTrain,
    SpikeTrainBatch,
    SpikeTypes,
    SpikeKeys
)

from x_io.axona.read_tetrode_and_cut import (
    load_spike_train_from_paths,
    _read_tetrode
)


from x_io.os_io_utils import (
    with_open_file
)
#from x_io.io_axona.tetrode_cut import _read_cut

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_cut_file_path = os.path.join(data_dir, '20140815-behavior2-90_1.cut')
test_tetrode_file_path = os.path.join(data_dir, '20140815-behavior2-90.1')

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
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return list(spk_time)

def make_2D_timestamps(count=20, T=2, dt=0.02):
    time = np.arange(0,T,dt)
    # spk_times = np.zeros((count, len(time)))
    spk_times = []

    for i in range(count):
        spk_count = np.random.choice(len(time), size=1)
        spk_times.append(np.random.choice(time, size=spk_count, replace=False).tolist())


    return list(spk_times)

############################

def test_spike_keys():
    spike_keys = SpikeKeys()

    spike_train_init_keys = spike_keys.get_spike_train_init_keys()

    assert type(spike_train_init_keys) == list
    for i in range(len(spike_train_init_keys)):
        assert type(spike_train_init_keys[i]) == str
    
def test_spike_types():
    spike_types = SpikeTypes()
    spike_keys = SpikeKeys()

    spike_train_init_keys = spike_keys.get_spike_train_init_keys()

    input_dict = spike_types.format_keys(spike_train_init_keys)
    keys = list(input_dict.keys())
    type_counter = [0,0,0]
    for i in range(len(keys)):
        if type(input_dict[keys[i]]) == int:
            type_counter[0] += 1
        if type(input_dict[keys[i]]) == float:
            type_counter[1] += 1
        if type(input_dict[keys[i]]) == list:
            type_counter[2] += 1

    assert type(input_dict) == dict
    assert sum(type_counter) == 4
    assert type_counter[-1] == 2



    

def test_spike_train_class():

    spike_keys = SpikeKeys()
    spike_types = SpikeTypes()

    spike_train_init_keys = spike_keys.get_spike_train_init_keys()
    input_dict = spike_types.format_keys(spike_train_init_keys)

    spike_times = make_1D_timestamps()

    T = 2
    dt = .02
    
    input_dict1 = input_dict.copy()
    input_dict1['sample_length'] = int(T / dt)
    input_dict1['sample_rate'] = float(T / dt)
    input_dict1['spikes_binary'] = []
    input_dict1['spike_times'] = spike_times

    spike_train1 = SpikeTrain(input_dict1)

    rate1 = spike_train1.get_spike_rate()
    spikes_binary1 = spike_train1.get_binary()

    assert type(rate1) == float
    assert type(spike_train1._spikes_binary) == list
    assert type(spike_train1._spike_times) == list
    assert type(spike_train1._spike_ids) == list

    spikes_binary2 = make_1D_binary_spikes()

    input_dict2 = input_dict.copy()
    input_dict2['sample_length'] = int(T / dt)
    input_dict2['sample_rate'] = float(T / dt)
    input_dict2['spikes_binary'] = spikes_binary2
    input_dict2['spike_times'] = []

    spike_train2 = SpikeTrain(input_dict2)

    rate2 = spike_train2.get_spike_rate()
    spike_times2 = spike_train2.get_spike_times()
    
    assert type(rate2) == float
    assert type(spike_train2._spikes_binary) == list
    assert type(spike_train2._spike_times) == list
    assert type(spike_train2._spike_ids) == list

def test_spike_train_batch_class():

    spike_keys = SpikeKeys()
    spike_types = SpikeTypes()

    spike_train_init_keys = spike_keys.get_spike_train_init_keys()
    input_dict = spike_types.format_keys(spike_train_init_keys)

    spike_times = make_2D_timestamps()

    T = 2
    dt = .02
    
    input_dict1 = input_dict.copy()
    input_dict1['sample_length'] = int(T / dt)
    input_dict1['sample_rate'] = float(T / dt)
    input_dict1['spikes_binary'] = []
    input_dict1['spike_times'] = spike_times

    spike_train1 = SpikeTrainBatch(input_dict1)
    rate1 = spike_train1.get_average_spike_rate()
    rate_list1 = spike_train1.get_indiv_spike_rate()
    spike_train1.get_binary()
    instances1 = spike_train1.get_spike_train_instances()

    assert type(rate1) == float
    assert type(rate_list1) == list
    assert type(spike_train1._spikes_binary) == list
    assert type(spike_train1._spike_times) == list
    assert type(spike_train1._spike_ids) == list
    assert type(spike_train1._spikes_binary[0]) == list
    assert type(spike_train1._spike_times[0]) == list
    assert isinstance(instances1[0], SpikeTrain) == True

    spikes_binary2 = make_2D_binary_spikes()

    input_dict2 = input_dict.copy()
    input_dict2['sample_length'] = int(T / dt)
    input_dict2['sample_rate'] = float(T / dt)
    input_dict2['spikes_binary'] = spikes_binary2
    input_dict2['spike_times'] = []

    spike_train2 = SpikeTrainBatch(input_dict2)

    spike_train2.get_spike_times()
    rate2 = spike_train2.get_average_spike_rate()
    rate_list2 = spike_train2.get_indiv_spike_rate()
    instances2 = spike_train2.get_spike_train_instances()

    assert type(rate2) == float
    assert type(rate_list2) == list
    assert type(spike_train2._spikes_binary) == list
    assert type(spike_train2._spike_times) == list
    assert type(spike_train2._spike_ids) == list
    assert type(spike_train2._spikes_binary[0]) == list
    assert type(spike_train2._spike_times[0]) == list
    assert isinstance(instances2[0], SpikeTrain) == True


if __name__ == '__main__':
    test_spike_keys()
    test_spike_types()
    test_spike_train_class()
    test_spike_train_batch_class()

