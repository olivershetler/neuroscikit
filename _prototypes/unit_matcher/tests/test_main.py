from email import header
import os
import sys

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

import pytest
# import _prototypes.unit_matcher.tests.read as read
from _prototypes.unit_matcher.main import match_session_units, _apply_remapping, _temp_read_cut, write_cut, _format_new_cut_file_name
from _prototypes.unit_matcher.read_axona import read_sequential_sessions

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'sessions': [session_settings,session_settings], 'smoothing_factor': 3}

session1, session2 = read_sequential_sessions(data_dir, settings_dict)

def test__format_new_cut_file_name():
    old_path = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    new_path = _format_new_cut_file_name(old_path)

    assert new_path == r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3_matched.cut'

def test__temp_read_cut():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  _temp_read_cut(open_cut_file)

    assert type(cut_data) == list 
    assert type(header_data) == list 
    assert type(cut_data[0]) == int

def test__apply_remapping():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    map_dict = {}
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  _temp_read_cut(open_cut_file)
    new_cut_data, header_data2 = _apply_remapping(cut_file, map_dict)

    assert header_data == header_data2
    # assert 

def test_write_cut():
    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3.cut'
    map_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:0, 8:0, 9:0, 10:0, 11:0}
    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data =  _temp_read_cut(open_cut_file)
    new_cut_data, header_data2 = _apply_remapping(cut_file, map_dict)

    cut_file = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\single_sequential\1-13_20210621-34-50x50cm-1500um-Test1_3_matched.cut'
    write_cut(cut_file, new_cut_data, header_data2)

    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data = _temp_read_cut(open_cut_file)

    assert new_cut_data == cut_data
    assert header_data == header_data2

def test_match_session_units():
    cut_file = match_session_units(session1, session2)

    with open(cut_file, 'r') as open_cut_file:
        cut_data, header_data = _temp_read_cut(open_cut_file)

    assert type(cut_data) == list 
    assert type(cut_data[0]) == int
    assert type(header_data) == list


# @pytest.mark.skip(reason="Not implemented yet")
# def test_read():
#     pass

# @pytest.mark.skip(reason="Not implemented yet")
# def test_write():
#     write_cut()
#     pass

# @pytest.mark.skip(reason="Not implemented yet")
# def test_main():
#     # read a file
#     # use a fixed re-mapping scheme
#     # write a new cut file
#     # read the new cut file
#     # compare the new cut file to the original
#     # assert that the new cut file conforms to the mapping scheme
#     pass


