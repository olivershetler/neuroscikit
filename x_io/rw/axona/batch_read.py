"""
This module contains methods for collecting, grouping and reading related data files of Axona formats.

This module will format the data into a dictionary that can be taken in by the Study bridge class.
"""

import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D
from library.study_space import Animal, Session, SessionData, SessionMetadata, Study, StudyMetadata
from core.instruments import DevicesMetadata, ImplantMetadata, TrackerMetadata
from core.subjects import AnimalMetadata
from core.spikes import Spike, SpikeTrain
from library.batch_space import SpikeTrainBatch, SpikeClusterBatch
import numpy as np

from x_io.rw.axona.read_tetrode_and_cut import (
    _format_spikes,
    _read_cut,
    _read_tetrode_header,
)

from x_io.rw.axona.read_pos import ( 
    grab_position_data,
)

def make_study(directory, settings_dict: list):

    if type(directory) != list:
        directory = [directory]
    
    # study_dict = _init_study_dict(settings_dict)
    study_dict = {}

    cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files(directory, pos_files=[], cut_files=[], tetrode_files=[], matched_cut_files=[], animal_dir_names=[])

    sorted_files, tetrode_counts, animal_ids = _group_session_files(cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names)

    assert len(sorted_files) == len(tetrode_counts)
    assert len(tetrode_counts) == len(animal_ids)

    indiv_session_settings = {}
    indiv_session_settings['tetrode_counts'] = tetrode_counts
    indiv_session_settings['animal_ids'] = animal_ids

    sessions = batch_sessions(sorted_files, settings_dict, indiv_session_settings)

    study_dict = _fill_study_dict(sessions, study_dict)

    study_metadata_dict = _get_study_metadata()
    study_metadata = StudyMetadata(study_metadata_dict)

    study = Study(study_metadata, study_dict)

    return study

def _get_study_metadata():
    return StudyMetadata({'test': 'test'})

def _grab_tetrode_cut_position_files(paths: list, pos_files=[], cut_files=[], tetrode_files=[], matched_cut_files=[], animal_dir_names=[], parent_path=None) -> tuple:

    '''
        Extract tetrode, cut, and position file data /+ a set file

        Params:
            files (list):
                List of file paths to tetrode, cut and position file OR folder directory
                containing all files

        Returns:
            Tuple: tetrode_files, cut_files, pos_files
            --------
            tetrode_files (list):
                List of all tetrode file paths
            cut_files (list):
                List of all cut file paths
            pos_files (list):
                List containing position file paths
    '''
 
    # Check for set file
    if len(paths) == 1 and os.path.isdir(paths[0]):
        files = os.listdir(paths[0])
        for file in files:
            fpath = paths[0] + '/' + file
            if os.path.isdir(fpath) and 'git' not in fpath:
                cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files(os.listdir(fpath), pos_files=pos_files, cut_files=cut_files, tetrode_files=tetrode_files, matched_cut_files=matched_cut_files, animal_dir_names=animal_dir_names, parent_path=fpath)
            if file[-3:] == 'pos':
                pos_files.append(fpath)
                animal_dir_names.append(os.path.basename(os.path.dirname(fpath)))
            elif file[-3:] == 'cut':
                if 'matched' not in file:
                    cut_files.append(fpath)
                else:
                    matched_cut_files.append(paths[0] + '/' + file)
            elif file[-1:].isdigit() and 'clu' not in file and 'cut' not in file and 'eeg' not in file and 'egf' not in file:
                tetrode_files.append(fpath)
    else:
        for file in paths:
            if parent_path != None:
                fpath = parent_path + '/' + file
                file = fpath
            if os.path.isdir(file) and 'git' not in file:
                cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names = _grab_tetrode_cut_position_files(os.listdir(file), pos_files=pos_files, cut_files=cut_files, tetrode_files=tetrode_files, matched_cut_files=matched_cut_files, animal_dir_names=animal_dir_names, parent_path=file)
            if file[-3:] == 'pos':
                pos_files.append(file)
                animal_dir_names.append(os.path.basename(os.path.dirname(fpath)))
            elif file[-3:] == 'cut':
                if 'matched' not in file:
                    cut_files.append( file)
                else:
                    matched_cut_files.append(file)
            elif file[-1:].isdigit() and 'clu' not in file and 'cut' not in file and 'eeg' not in file and 'egf' not in file:
                tetrode_files.append(file)

    return cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names

def _group_session_files(cut_files, tetrode_files, pos_files, matched_cut_files, animal_dir_names):

    '''
        Group position, cut and tetrode files that belong to the same session.

        Params:
            pos_files, tetrode_files, cut_files (list):
                Lists of position file, tetrode_files, and cut_files respectively

        Returns:
            grouped_session (list):
                A nested list where each element is a collection of cut, tetrode and pos files
                belonging to the same session
    '''

    # Initialize empty list where each element will hold common session files
    grouped_sessions = []
    tetrode_counts = []
    animal_ids = []

    c = 0

    # Iterate over each position file
    for pos_file in pos_files:
        # Will group common files
        collection = {}

        # Grab session handle from .pos file
        session = pos_file[:-4]

        # Grab the tetrode and cut files belonging to this session only
        select_tetrodes = [tetrode for tetrode in tetrode_files if session in tetrode]
        select_cuts = [cut for cut in cut_files if session in cut]
        select_cuts_matched = [cut for cut in matched_cut_files if session in cut]

        # Ensures that only collections containing all three: pos, cut and tetrode files are added
        if len(select_tetrodes) == 0 or len(select_cuts) == 0:
            break

        # Add these files into a single data strucutre
        # collection.append(pos_file)
        # collection += (select_tetrodes + select_cuts + select_cuts_matched)
        collection['pos'] = pos_file
        collection['tet'] = sorted(select_tetrodes)
        collection['cut'] = sorted(select_cuts)
        collection['matched_cut'] = sorted(select_cuts_matched)

        # Accumulate these collections to a separate data structure as 'groups' of files.
        grouped_sessions.append(collection)
        tetrode_counts.append(len(select_tetrodes))
        animal_ids.append(animal_dir_names[c])

        c += 1

    return grouped_sessions, tetrode_counts, animal_ids

def _init_study_dict(settings_dicts):

    study_dict = {}

    for i in range(len(settings_dicts['sessions'])):
        study_dict['session_' + str(i+1)] = {}

    return study_dict

def _fill_study_dict(sessions, study_dict):
    # assert len(sessions) == len(study_dict)

    for i in range(len(sessions)):
         study_dict['session_' + str(i+1)] = sessions['session_' + str(i+1)]

    return study_dict

def batch_sessions(sorted_files, settings_dict, indiv_session_settings):
    """
    Sorted files: A nested list where each element is a collection of cut, tetrode and pos files
    belonging to the same session
    
    """
    # assert len(cut_files) == len(tet_files)
    # assert len(tet_files) == len(pos_files)
    # assert len(sorted_files) == len(settings_dict['sessions'])

    sessions = {}

    c = 1

    for i in range(len(sorted_files)):


        pos_file = sorted_files[i]['pos']
        cut_files = sorted_files[i]['cut']
        tet_files = sorted_files[i]['tet']
        matched_cut_files = sorted_files[i]['matched_cut']

        assert len(cut_files) == len(tet_files), "Number of tetrode and cut files doesn't match"

        for j in range(len(cut_files)):

            session_settings_dict = settings_dict['session']
            # session_settings_dict['channel_count'] = indiv_session_settings['tetrode_counts'][i]
            animal_id = str(indiv_session_settings['animal_ids'][i] + '_tet' + str(j+1))

            session_settings_dict['animal'] = {'animal_id': animal_id}
            

            if settings_dict['useMatchedCut'] == True: 
                assert len(sorted_files[i]) > 3, print('Matched cut file not present, make sure to run unit matcher')
                cut_file = matched_cut_files[j]
                assert 'matched.cut' in cut_file
            else:
                cut_file = cut_files[j]

            tet_file = tet_files[j]

            session = make_session(cut_file, tet_file, pos_file, settings_dict, session_settings_dict)
            # session = make_session(sorted_files[i], session_settings_dict, settings_dict['ppm'])

            session.set_smoothing_factor(settings_dict['smoothing_factor'])

            sessions['session_'+str(c)] = session

            c += 1

    return sessions


def make_session(cut_file, tet_file, pos_file, settings_dict, session_settings_dict):

    ppm = settings_dict['ppm']

    session_dict = _init_session_dict(session_settings_dict)

    implant_data_dict, ch_count = _get_session_data(cut_file, tet_file, ch_count=session_settings_dict['channel_count'])

    session_settings_dict['channel_count'] = ch_count

    if session_settings_dict['devices']['axona_led_tracker'] == True:
        pos_dict = grab_position_data(pos_file, ppm)
        implant_data_dict['sample_rate'] = pos_dict['sample_rate']

    session_dict = _fill_session_dict(session_dict, implant_data_dict, pos_dict, session_settings_dict)

    session, session_classes = _create_session_classes(session_dict, session_settings_dict)

    # session.set_animal_id()

    session.session_metadata.add_file_paths(cut_file, tet_file, pos_file, ppm)

    assert isinstance(session, Session)

    return session

def _create_session_classes(session_dict, settings_dict):
    """
    Tetrode, Cut, Pos, Animal metadata, device metadata

    For one cut, tet, pos file:
        - extract tet data --> spikeTrain(event_times), Spike(events)
        - extract pos data --> locationClass(x,y)
        - animal metadata --> animal()
        - devices metadata --> tracker(devices), implant(devices)
        - arena metadata?

    ALL TAKE IN SESSION METADATA
    """


    session = Session()

    animal_metadata = session.make_class(AnimalMetadata, session_dict['animal'])
    # session.set_animal_id()

    tracker_dict = {}
    for key in session_dict['devices']['axona_led_tracker']:
        if 'data' not in str(key):
            tracker_dict[key] = session_dict['devices']['axona_led_tracker'][key]
    tracker_metadata = session.make_class(TrackerMetadata, tracker_dict)

    implant_dict = {}
    for key in session_dict['devices']['implant']:
        if 'data' not in str(key):
            implant_dict[key] = session_dict['devices']['implant'][key]
    implant_metadata = session.make_class(ImplantMetadata, implant_dict)

    spike_cluster = session.make_class(SpikeClusterBatch, session_dict['devices']['implant']['implant_data'])
    spike_train = session.make_class(SpikeTrain, session_dict['devices']['implant']['implant_data'])
    position = session.make_class(Position2D, session_dict['devices']['axona_led_tracker']['led_position_data'])
    
    # animal_metadata = AnimalMetadata(session_dict['animal'])
    # tracker_metadata = TrackerMetadata(session_dict['devices']['implant'])
    # implant_metadata = ImplantMetadata(session_dict['devices']['axona_led_tracker'])
    # devices_dict = {'axona_led_tracker': tracker_metadata, 'implant': implant_metadata}
    # devices_metadata = DevicesMetadata(devices_dict)
    
    # spike_train = SpikeTrain(session_dict['devices']['implant']['implant_data'])
    # spike_cluster = SpikeClusterBatch(session_dict['devices']['implant']['implant_data'])

    # # Temporary fix, rmeove this
    # position = Position2D('subject' , 'space', session_dict['devices']['axona_led_tracker'])

    # # Workspace classes
    # session_metadata = SessionMetadata({'animal': animal_metadata, 'devices': devices_metadata})
    # session_data = SessionData({'spike_train': spike_train, 'spike_cluster': spike_cluster, 'position': position})

    session_classes = {'metadata': session.session_metadata, 'data': session.session_data}

    return session, session_classes 



def _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict):
    devices = settings_dict['devices']

    if devices['axona_led_tracker'] == True:
        session_dict['devices']['axona_led_tracker']['led_position_data'] = pos_dict

    if devices['implant'] == True:
        session_dict['devices']['implant']['implant_data'] = implant_data_dict

    animal = settings_dict['animal']
    session_dict['animal'] = animal

    return session_dict

def _init_session_dict(settings_dict):
    session_dict = {}
    session_dict['animal'] = {}
    session_dict['devices'] = {}
 
    animal_keys = settings_dict['animal']
    # animal_keys = ['animal_id', 'species', 'sex', 'age', 'weight', 'genotype', 'animal_notes'] 
    devices = settings_dict['devices']
    # e.g. ['axona_led_tracker': True, 'implant': True,]
    implant = settings_dict['implant']
    # [implant id, type, geometry, wire length, wire length units, implant units]

    for key in animal_keys:
        if key == 'age' or key == 'weight':
            session_dict['animal'][key] = float
        else:
            session_dict['animal'][key] = str

    for key in devices:
        if devices[key] == True:
            session_dict['devices'][key]= {}

    for key in implant:
        session_dict['devices']['implant'][key] = implant[key]

    session_dict['devices']['implant']['implant_data'] = {}   

    # session_dict['devices']['implant']['implant_id'] = implant['implant_id']
    # session_dict['devices']['implant']['implant_type'] = implant['implant_type']
    # session_dict['devices']['implant']['implant_geometry'] = implant['implant_geometry']
    # session_dict['devices']['implant']['wire_length'] = implant['wire_length']
    # session_dict['devices']['implant']['wire_length_units'] = implant['wire_length_units']
    # session_dict['devices']['implant']['implant_units'] = implant['implant_units']
    # session_dict['devices']['implant']['implant_data'] = {}   

    return session_dict

def _get_session_data(cut_file, tet_file, ch_count=4):

    with open(cut_file, 'r') as open_cut_file, open(tet_file, 'rb') as open_tet_file:
        cut_data = _read_cut(open_cut_file)
        tetrode_data = _format_spikes(open_tet_file)

    if ch_count != len(tetrode_data[1]):
        ch_count = len(tetrode_data[1])

    implant_data_dict = _init_implant_data(ch_count)

    implant_data_dict = _fill_implant_data(implant_data_dict, tetrode_data, cut_data, ch_count)

    return implant_data_dict, ch_count

def _fill_implant_data(implant_data_dict, tetrode_data, cut_data, ch_count):
    implant_data_dict['duration'] = tetrode_data[-1]['duration']
    # implant_data_dict['sample_rate'] = tetrode_data[-1]['sample_rate']

    implant_data_dict['datetime'] = tetrode_data[-1]['datetime']

    for ch in range(ch_count):
        implant_data_dict['channel_'+str(ch+1)] = tetrode_data[1]['ch'+str(ch+1)].tolist()

    implant_data_dict['event_times'] = tetrode_data[0].tolist()
    implant_data_dict['event_labels'] = cut_data

    return implant_data_dict

def _init_implant_data(ch_count):
    implant_data_dict = {}

    ch_keys = []
    for ch in range(ch_count):
        key = 'channel_' + str(ch+1)
        ch_keys.append(key)

    keys = ['event_times', 'event_labels', 'sample_rate', 'duration']

    keys = np.hstack((keys, ch_keys))

    for key in keys:
        if key == 'sample_rate':
            implant_data_dict[key] = float
        if key == 'duration':
            implant_data_dict[key] = float
        else:
            implant_data_dict[key] = []

    return implant_data_dict



