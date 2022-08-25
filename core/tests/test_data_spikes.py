import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.data_spikes import (
    SpikeTrain
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
        spike_trains[i] = np.random.randint(2, size=size)

    return list(spike_trains)

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False)

    return list(spk_time)

def make_2D_timestamps(count=20, T=2, dt=0.02):
    time = np.arange(0,T,dt)
    spk_times = np.zeros((count, len(time)))

    for i in range(count):
        spk_count = np.random.choice(len(time), size=1)
        spk_times[i] = list(np.random.choice(time, size=spk_count, replace=False))


    return list(spk_times)

def test_spike_train_init():

    spike_times = make_1D_timestamps()
    T = 2
    dt = .02
    time = np.arange(0,T,dt)

    spike_train = SpikeTrain(time, spike_times=spike_times)

    assert type(spike_train.spikes_raw) == list
    assert type(spike_train.spike_times) == list
    assert type(spike_train.spike_ids) == list
    assert type(spike_train.spike_features) == list

    spike_ids, spike_times, spikes_raw = spike_train.__getitem__(5)

    spikes_raw = make_1D_binary_spikes()

    spike_train = SpikeTrain(time, spikes_raw=spikes_raw)

    assert type(spike_train.spikes_raw) == list
    assert type(spike_train.spike_times) == list
    assert type(spike_train.spike_ids) == list
    assert type(spike_train.spike_features) == list

    spike_ids, spike_times, spikes_raw = spike_train.__getitem__(5)

def test_spike_train_rate_functions():

    spike_times = make_1D_timestamps()
    T = 2
    dt = .02
    time = np.arange(0,T,dt)

    spike_train = SpikeTrain(time, spike_times=spike_times)

    rate1 = spike_train.spike_rate()

    assert type(rate1) == float

    spikes_raw = make_1D_binary_spikes()

    spike_train = SpikeTrain(time, spikes_raw=spikes_raw)

    rate2 = spike_train.spike_rate()
    
    assert type(rate2) == float







    
if __name__ == '__main__':
    test_spike_train_rate_functions()

