import os
import sys
import numpy as np

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

# import _prototypes.unit_matcher.tests.read as read
from _prototypes.unit_matcher.main import format_cut, run_unit_matcher, map_unit_matches_first_session, map_unit_matches_sequential_session
from _prototypes.unit_matcher.read_axona import read_sequential_sessions, temp_read_cut
from _prototypes.unit_matcher.session import compare_sessions
from _prototypes.unit_matcher.write_axona import format_new_cut_file_name

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential'
print(">>>>>", data_dir)

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'sessions': [session_settings,session_settings], 'smoothing_factor': 3, 'useMatchCut': False}

session1, session2 = read_sequential_sessions(data_dir, settings_dict)

matches, match_distances, unmatched_2, unmatched_1 = compare_sessions(session1, session2)

def test_map_unit_matches_first_session():
    map_dict = map_unit_matches_first_session(matches, match_distances, unmatched_1)
    new_ids = sorted(list(map_dict.values()))

    assert min(new_ids) == 1
    assert len(np.arange(min(new_ids), max(new_ids)+1, 1)) == len(new_ids) + 1 or len(np.arange(min(new_ids), max(new_ids)+1, 1)) == len(new_ids)

def test_map_unit_matches_sequential_session():
    map_dict = map_unit_matches_sequential_session(matches, unmatched_2)
    new_ids = sorted(list(map_dict.values()))

    assert min(new_ids) == 1
    assert len(np.arange(min(new_ids), max(new_ids)+1, 1)) == len(new_ids) + 1 or len(np.arange(min(new_ids), max(new_ids)+1, 1)) == len(new_ids)

def test_format_cut():
    map_dict = map_unit_matches_first_session(matches, match_distances, unmatched_1)
    new_cut_file_path, new_cut_data, header_data = format_cut(session1, map_dict)

    assert 'matched' in new_cut_file_path
    assert type(new_cut_data) == list
    assert type(header_data) == list

def test_run_unit_matcher():
    study = run_unit_matcher([data_dir], settings_dict)
    for ses in study.sessions:
        cut_file_path = ses.session_metadata.file_paths['cut']
        new_cut_file_path = format_new_cut_file_name(cut_file_path)
        with open(new_cut_file_path, 'r') as open_cut_file:
            cut_values, _ =  temp_read_cut(open_cut_file)
        cut_values = np.unique(cut_values)
        valid = []
        for val in sorted(cut_values):
            if val != 0:
                if val >=1:
                    valid.append(val)
                if val + 1 not in cut_values:
                    break
        assert len(np.unique(ses.get_cell_data()['cell_ensemble'].get_label_ids())) == len(valid) or len(np.unique(ses.get_cell_data()['cell_ensemble'].get_label_ids())) == len(valid) + 1






# def test_match_session_units():
#     cut_file = match_session_units(session1, session2)

#     with open(cut_file, 'r') as open_cut_file:
#         cut_data, header_data = temp_read_cut(open_cut_file)

#     assert type(cut_data) == list
#     assert type(cut_data[0]) == int
#     assert type(header_data) == list

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


