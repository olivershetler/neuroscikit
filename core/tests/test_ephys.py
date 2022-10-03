import os
from random import sample
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.ephys import (
    EphysCollection,
    EphysSeries,
)

from x_io.rw.intan.load_intan_rhd_format.load_intan_rhd_format import read_rhd_data
from x_io.rw.intan.read_rhd import read_rhd

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_rhd_file_path = os.path.join(data_dir, 'sampledata.rhd')

np.random.seed(0)

def test_ephys_series():
    collection = read_rhd(test_rhd_file_path)
    ephys_series = collection.signal['channel_0']

    data = ephys_series.signal
    prev = len(data)

    sample_rate = ephys_series.sample_rate[0]

    ephys_series.down_sample(sample_rate/2)

    ephys_series.add_filtered([], method='test', type='test')

    filtered_dict = ephys_series.get_filtered_dict()

    assert 'test' in filtered_dict.keys()
    assert 'test' in filtered_dict['test'].keys()
    assert len(ephys_series.signal) == prev/2
    assert type(ephys_series.filtered) == list

def test_ephys_collection():
    collection = read_rhd(test_rhd_file_path)

    power_bands = collection.get_power_bands()
    filtered = collection.get_filtered()

    assert type(collection.num_channels) == int
    assert len(collection.signal.keys()) == collection.num_channels
    assert type(filtered) == list
    assert type(filtered[0]) == list
    assert type(power_bands) == list
    assert type(power_bands[0]) == list

    filtered = collection.get_filtered(method='notch_filt', type='cheby2')

    assert type(filtered) == list
    assert type(filtered[0]) == list


if __name__ == '__main__':
    test_ephys_series()
    test_ephys_collection()
    print('we good')
