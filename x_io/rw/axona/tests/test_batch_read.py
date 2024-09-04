
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online
import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from x_io.rw.axona.read_tetrode_and_cut import (
    _read_cut
    ,_read_tetrode_header
    ,_read_tetrode
    ,_format_spikes
    ,get_spike_trains_from_channel
    ,load_spike_train_from_paths
)

from library.study_space import Session, SessionData, SessionMetadata, Study, StudyMetadata
from core.instruments import DevicesMetadata, ImplantMetadata, TrackerMetadata
from core.subjects import AnimalMetadata
from core.spikes import SpikeTrain
from library.batch_space import SpikeClusterBatch

from x_io.rw.axona.read_pos import (
    grab_position_data,
)

from x_io.rw.axona.batch_read import (
    _init_implant_data,
    _fill_implant_data,
    _get_session_data,
    _init_session_dict,
    _fill_session_dict,
    make_session,
    _grab_tetrode_cut_position_files,
    _init_study_dict,
    _group_session_files,
    batch_sessions,
    make_study,

)

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data/test_dir')
cut_file = os.path.join(data_dir, '20140815-behavior2-90_1.cut')
tet_file = os.path.join(data_dir, '20140815-behavior2-90.1')
pos_file = os.path.join(data_dir, '20140815-behavior2-90.pos')

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': False, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}


settings_dict = {'ppm': 511, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': False}


def test__init_implant_data():
    ch_count = 4
    implant_data_dict = _init_implant_data(ch_count)

    assert type(implant_data_dict) == dict

    keys = ['event_times', 'event_labels', 'sample_rate', 'duration']

    for key in keys:
        assert key in implant_data_dict

    keys = implant_data_dict.keys()
    for i in range(ch_count):
        assert 'channel_' + str(i+1) in keys

def test__fill_implant_data():
    ch_count = 4

    implant_data_dict = _init_implant_data(ch_count)

    with open(cut_file, 'r') as open_cut_file, open(tet_file, 'rb') as open_tet_file:
        cut_data = _read_cut(open_cut_file)
        tetrode_data = _format_spikes(open_tet_file)

    implant_data_dict = _fill_implant_data(implant_data_dict, tetrode_data, cut_data, ch_count)

    assert type(implant_data_dict) == dict

    keys = ['event_times', 'event_labels', 'sample_rate', 'duration']

    for key in keys:
        if key != 'sample_rate' and key != 'duration':
            assert type(implant_data_dict[key]) == list
        elif key == 'sample_rate' and key == 'duration':
            assert type(implant_data_dict[key]) == float

def test__get_session_data():
    ch_count = 4

    implant_data_dict, ch_count = _get_session_data(cut_file, tet_file, ch_count)

    assert type(implant_data_dict) == dict

    keys = ['event_times', 'event_labels', 'sample_rate', 'duration']

    for key in keys:
        if key != 'sample_rate' and key != 'duration':
            assert type(implant_data_dict[key]) == list
        elif key == 'sample_rate' and key == 'duration':
            assert type(implant_data_dict[key]) == float

def test__init_session_dict():
    session_dict = _init_session_dict(settings_dict['session'])

    assert type(session_dict) == dict
    assert len(session_dict['devices']['implant']) == 7
    assert 'animal' in session_dict

def test__fill_session_dict():
    session_dict = _init_session_dict(settings_dict['session'])

    if devices['axona_led_tracker'] == True:
        pos_dict = grab_position_data(pos_file, ppm=settings_dict['ppm'])
    else: 
        pos_dict = {}

    ch_count = 4

    implant_data_dict, ch_count = _get_session_data(cut_file, tet_file, ch_count)

    session_dict = _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict['session'])
    
    if devices['axona_led_tracker'] == True:
        assert 'x' in session_dict['devices']['axona_led_tracker']['led_position_data']
    if devices['implant'] == True:
        assert 'event_times' in session_dict['devices']['implant']['implant_data']

def test_make_session():

    session = make_session(cut_file, tet_file, pos_file, settings_dict, settings_dict['session'])

    assert isinstance(session, Session)
    if devices['axona_led_tracker'] == True:
        assert isinstance(session.get_devices_metadata()['axona_led_tracker'], TrackerMetadata)
    if devices['implant'] == True:
        assert isinstance(session.get_devices_metadata()['implant'], ImplantMetadata)
    assert isinstance(session.get_spike_data()['spike_train'], SpikeTrain)
    assert isinstance(session.get_spike_data()['spike_cluster'], SpikeClusterBatch)
    assert int(session.get_spike_data()['spike_cluster'].waveform_sample_rate) == int(48000)

def test__grab_tetrode_cut_position_files():

    cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files([data_dir], pos_files=[], cut_files=[], tetrode_files=[])

    assert type(cut_files) == list
    assert type(pos_files) == list
    assert type(tetrode_files) == list
    assert type(matched_cut_files) == list
    assert type(animal_dir_names) == list

    # assert len(cut_files) == 1
    # assert len(pos_files) == 1
    # assert len(tetrode_files) == 1
    # assert len(matched_cut_files) == 0
    # assert len(animal_dir_names) == 1

# def test__init_study_dict():
#     study_dict = _init_study_dict(settings_dict)

#     assert type(study_dict) == dict

def test__group_session_files():

    cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files([data_dir], pos_files=[], cut_files=[], tetrode_files=[])

    sorted_files, tetrode_counts, animal_ids = _group_session_files(cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names)

    assert len(sorted_files[0]['cut']) == len(cut_files)
    assert len(cut_files) == len(tetrode_files)

def test_batch_sessions():

    cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files([data_dir], pos_files=[], cut_files=[], tetrode_files=[])
    sorted_files, tetrode_counts, animal_ids = _group_session_files(cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names)
    indiv_session_settings = {}
    indiv_session_settings['tetrode_counts'] = 1
    indiv_session_settings['animal_ids'] = '1'
    sessions = batch_sessions(sorted_files, settings_dict, indiv_session_settings)

    assert type(sessions) == dict
    assert len(sessions) == 1
    assert isinstance(sessions['session_1'], Session)

def test_make_study():
    study = make_study([data_dir], settings_dict)

    assert isinstance(study, Study)
    assert len(study.sessions) > 0
    assert isinstance(study._input_dict['session_1'], Session)



