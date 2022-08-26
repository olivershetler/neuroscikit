import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.core_utils import (
    make_seconds_index_from_rate, 
    make_hms_index_from_rate, 
    SpikeKeys, 
    SpikeTypes
)

from datetime import datetime

def _get_seconds(string_time):
    return float(string_time.split(':')[-1])

def test_make_time_index_from_rate():
    for n_samples in [2, 15, 40, 200]:
        for sample_rate in [50, 250, 1000, 10000]:
            time_index = make_hms_index_from_rate('09:15:56', n_samples, sample_rate)
            start = _get_seconds(time_index[0])
            second = _get_seconds(time_index[1])
            end = _get_seconds(time_index[-1])
            r = lambda n: round(n, 6)
            assert r(second - start) == r(1/sample_rate)
            assert r(end - start) == r((n_samples-1)/sample_rate)

def test_make_seconds_index_from_rate():
    for n_samples in [2, 15, 40, 200]:
        for sample_rate in [50, 250, 1000, 10000]:
            time_index = make_seconds_index_from_rate(n_samples, sample_rate)
            start = time_index[0]
            second = time_index[1]
            end = time_index[-1]
            r = lambda n: round(n, 6)
            assert r(second - start) == r(1/sample_rate)
            assert r(end - start) == r((n_samples-1)/sample_rate)

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




if __name__ == '__main__':
    test_make_time_index_from_rate()
    test_make_seconds_index_from_rate()
    test_spike_keys()
    test_spike_types()