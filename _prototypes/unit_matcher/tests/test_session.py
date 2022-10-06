import os
import sys
import numpy as np

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

from library.study_space import Session
from _prototypes.unit_matcher.read_axona import read_sequential_sessions
from _prototypes.unit_matcher.session import compare_sessions, compute_distances, guess_remaining_matches, extract_full_matches, map_unit_matches

data_dir = parent_dir + r'\neuroscikit_test_data\single_sequential'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'sessions': [session_settings,session_settings], 'smoothing_factor': 3}

session1, session2 = read_sequential_sessions(data_dir, settings_dict)

def test_map_unit_matches():
    matches = [[1,2], [2,3], [3,4]] 

    map_dict = map_unit_matches(matches)

    assert map_dict[2] == 1
    assert map_dict[3] == 2
    assert map_dict[4] == 3


def test_compute_distances():
    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])

    assert type(distances) == np.ndarray
    assert type(pairs) == np.ndarray
    assert len(distances) == len(pairs)
    assert len(np.unique(session1.get_cell_data()['cell_ensemble'].cluster_labels)) == len(distances)
    assert len(np.unique(session2.get_cell_data()['cell_ensemble'].cluster_labels)) == len(distances.T)

def test_extract_full_matches():
    # distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])

    distances = np.array([[5,1,5], [5,1,5], [5,5,2]])
    pairs = np.array([[[0,0],[0,1],[0,2]], [[1,0],[1,1],[1,2]], [[2,0], [2,1], [2,2]]])

    full_matches, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)

    assert type(full_matches) == list
    assert type(remaining_distances) == np.ndarray
    assert type(remaining_pairs) == np.ndarray

    assert [1,0].all() in full_matches 
    assert [2,2].all() in full_matches

def test_guess_remaining_matches():
    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])

    full_matches, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)
    
    remaining_matches, unmmatched = guess_remaining_matches(remaining_distances, remaining_pairs)

    assert type(remaining_matches) == list
    assert type(unmmatched) == list
    assert len(full_matches) + len(remaining_matches) + len(unmmatched) == len(distances)


def test_compare_sessions():
    map_dict = compare_sessions(session1, session2)

    assert type(map_dict) == dict 
    assert len(np.unique(session2.get_cell_data()['cell_ensemble'].cluster_labels)) == len(map_dict)