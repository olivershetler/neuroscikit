"""
This module contains methods for collecting, grouping and reading related data files of Axona formats.

This module will format the data into a dictionary that can be taken in by the Study bridge class.
"""

import os, sys
from xml.etree.ElementTree import TreeBuilder

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import numpy as np

from x_io.rw.axona.read_tetrode_and_cut import (
    _format_spikes,
    _read_cut,
    _read_tetrode_header,
)

from x_io.rw.axona.read_pos import ( 
    grab_position_data,
)

def make_study_dict(directory, settings_dict: list):

    study_dict = _init_study_dict(settings_dict)

    cut_files, tetrode_files, pos_files = _grab_tetrode_cut_position_files(directory, pos_files=[], cut_files=[], tetrode_files=[])

    sorted_files = _group_session_files(cut_files, tetrode_files, pos_files)

    session_dicts = batch_sessions(sorted_files, settings_dict)

    study_dict = _fill_study_dict(session_dicts, study_dict)

    return study_dict

def _grab_tetrode_cut_position_files(paths: list, pos_files=[], cut_files=[], tetrode_files=[], parent_path=None) -> tuple:

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
                tetrode_files, cut_files, pos_files = _grab_tetrode_cut_position_files(os.listdir(fpath), pos_files=pos_files, cut_files=cut_files, tetrode_files=tetrode_files, parent_path=fpath)
            if file[-3:] == 'pos':
                pos_files.append(paths[0] + "/" + file)
            elif file[-3:] == 'cut':
                cut_files.append(paths[0] + "/" + file)
            elif file[-1:].isdigit() and 'clu' not in file:
                tetrode_files.append(paths[0] + "/" + file)
    else:
        for file in paths:
            if parent_path != None:
                fpath = parent_path + '/' + file
                file = fpath
            if os.path.isdir(file) and 'git' not in file:
                tetrode_files, cut_files, pos_files = _grab_tetrode_cut_position_files(os.listdir(file), pos_files=pos_files, cut_files=cut_files, tetrode_files=tetrode_files, parent_path=file)
            if file[-3:] == 'pos':
                pos_files.append(file)
            elif file[-3:] == 'cut':
                cut_files.append(file)
            elif file[-1:].isdigit() and 'clu' not in file:
                tetrode_files.append(file)
    return cut_files, tetrode_files, pos_files

def _group_session_files(cut_files, tetrode_files, pos_files):

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

    # Iterate over each position file
    for pos_file in pos_files:
        # Will group common files
        collection = []

        # Grab session handle from .pos file
        session = pos_file[:-4]

        # Grab the tetrode and cut files belonging to this session only
        select_tetrodes = [tetrode for tetrode in tetrode_files if session in tetrode]
        select_cuts = [cut for cut in cut_files if session in cut]

        # Ensures that only collections containing all three: pos, cut and tetrode files are added
        if len(select_tetrodes) == 0 or len(select_cuts) == 0:
            break

        # Add these files into a single data strucutre
        collection.append(pos_file)
        collection += (select_tetrodes + select_cuts)

        # Accumulate these collections to a separate data structure as 'groups' of files.
        grouped_sessions.append(collection)

    return grouped_sessions

def _init_study_dict(settings_dicts):

    study_dict = {}

    for i in range(len(settings_dicts['sessions'])):
        study_dict['session_' + str(i+1)] = {}

    return study_dict

def _fill_study_dict(session_dicts, study_dict):
    assert len(session_dicts) == len(study_dict)

    for i in range(len(session_dicts)):
         study_dict['session_' + str(i+1)] = session_dicts[i]

    return study_dict

def batch_sessions(sorted_files, settings_dict):
    """
    Sorted files: A nested list where each element is a collection of cut, tetrode and pos files
    belonging to the same session
    
    """
    # assert len(cut_files) == len(tet_files)
    # assert len(tet_files) == len(pos_files)
    assert len(sorted_files) == len(settings_dict['sessions'])

    session_dicts = []

    for i in range(len(settings_dict['sessions'])):

        session_dict = make_session_dict(sorted_files[i][2], sorted_files[i][1], sorted_files[i][0], settings_dict['sessions'][i], settings_dict['ppm'])

        session_dicts.append(session_dict)

    return session_dicts




     
def make_session_dict(cut_file, tet_file, pos_file, settings_dict, ppm):
    session_dict = _init_session_dict(settings_dict)

    implant_data_dict = _get_session_data(cut_file, tet_file, ch_count=settings_dict['channel_count'])

    if settings_dict['devices']['axona_led_tracker'] == True:
        pos_dict = grab_position_data(pos_file, ppm)

    session_dict = _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict)

    return session_dict

def _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict):
    devices = settings_dict['devices']

    if devices['axona_led_tracker'] == True:
        session_dict['devices']['axona_led_tracker'] = pos_dict

    if devices['implant'] == True:
        session_dict['devices']['implant']['implant_data'] = implant_data_dict
    
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

    session_dict['devices']['implant']['implant_id'] = implant['implant_id']
    session_dict['devices']['implant']['implant_type'] = implant['implant_type']
    session_dict['devices']['implant']['implant_geometry'] = implant['implant_geometry']
    session_dict['devices']['implant']['wire_length'] = implant['wire_length']
    session_dict['devices']['implant']['wire_length_units'] = implant['wire_length_units']
    session_dict['devices']['implant']['implant_units'] = implant['implant_units']
    session_dict['devices']['implant']['implant_data'] = {}   

    return session_dict

def _get_session_data(cut_file, tet_file, ch_count=4):

    with open(cut_file, 'r') as open_cut_file, open(tet_file, 'rb') as open_tet_file:
        cut_data = _read_cut(open_cut_file)
        tetrode_data = _format_spikes(open_tet_file)

    implant_data_dict = _init_implant_data(ch_count)

    implant_data_dict = _fill_implant_data(implant_data_dict, tetrode_data, cut_data, ch_count)

    return implant_data_dict

def _fill_implant_data(implant_data_dict, tetrode_data, cut_data, ch_count):
    implant_data_dict['duration'], implant_data_dict['sample_rate'] = tetrode_data[-1]['duration'], tetrode_data[-1]['sample_rate']

    for ch in range(ch_count):
        implant_data_dict['channel_'+str(ch+1)] = tetrode_data[ch+1].tolist()

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



