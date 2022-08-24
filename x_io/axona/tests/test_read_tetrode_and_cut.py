
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

#import urllib as web
import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from x_io.axona.read_tetrode_and_cut import (
    _read_cut
    ,_read_tetrode_header
    ,_read_tetrode
    ,_format_spikes
    ,get_spike_trains_from_channel
    ,load_spike_train_from_paths
)

from x_io.os_io_utils import (
    with_open_file
)
#from x_io.io_axona.tetrode_cut import _read_cut

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_cut_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90_1.cut')
test_tetrode_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90.1')

def test__read_cut():

    with open(test_cut_file_path, 'r') as cut_file:
        cut_values = _read_cut(cut_file)
    assert type(cut_values) == list

def test__read_tetrode_header():
    with open(test_tetrode_file_path, 'rb') as tetrode_file:
        tetrode_header = _read_tetrode_header(tetrode_file)
    assert type(tetrode_header) == dict
    print(tetrode_header)

def test__read_tetrode():
    with open(test_tetrode_file_path, 'rb') as tetrode_file:
        tetrode_values = _read_tetrode(tetrode_file)
    #tetrode_values = _read_tetrode(test_tetrode_file_path)
    #print(tetrode_values)
    assert type(tetrode_values) == tuple
    assert type(tetrode_values[0]) == dict
    assert type(tetrode_values[1]) == dict

def test__format_spikes():
    with open(test_tetrode_file_path, 'rb') as tetrode_file:
        tetrode_values = _format_spikes(tetrode_file)
    assert type(tetrode_values) == tuple


def test_get_spike_trains_from_channel():
    with open(test_cut_file_path, 'r') as cut_file, open(test_tetrode_file_path, 'rb') as tetrode_file:
        channel, empty_cell = get_spike_trains_from_channel(cut_file, tetrode_file, 1)
    assert type(channel) == list
    assert type(empty_cell) == int
    assert len(channel) == empty_cell


def test_load_spike_train_from_paths():
    channel, empty_cell = load_spike_train_from_paths(test_cut_file_path, test_tetrode_file_path, 1)



