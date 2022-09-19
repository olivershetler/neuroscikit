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


def batch_sessions(cut_files, tet_files, pos_files, settings_dicts: list):
    """
    List of files and list of settings dicts for each session
    
    """
    assert len(cut_files) == len(tet_files)
    assert len(tet_files) == len(pos_files)
    assert len(pos_files) == len(settings_dicts)

    session_dicts = []

    for i in range(len(cut_files)):

        session_dict = make_session_dict(cut_files[i], tet_files[i], pos_files[i], settings_dicts[i])

        session_dicts.append(session_dict)

    return session_dicts

       
def make_session_dict(cut_file, tet_file, pos_file, settings_dict):
    session_dict = _init_session_dict(settings_dict)

    implant_data_dict = _format_session(cut_file, tet_file, ch_count=settings_dict['channel_count'])

    if settings_dict['devices']['axona_led_tracker'] == True:
        pos_dict = grab_position_data(pos_file)

    session_dict = _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict)

    return session_dict



def _fill_session_dict(session_dict, implant_data_dict, pos_dict, settings_dict):
    devices = settings_dict['devices']

    if devices['axona_led_tracker'] == True:
        session_dict['devices']['axona_led_tracker'] = pos_dict

    if devices['implant'] == True:
        session_dict['devices']['implants']['implant_data'] = implant_data_dict
    
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
    # [implant id, type, geometry]

    for key in animal_keys:
        if key == 'age' or key == 'weight':
            session_dict['animal'][key] = float
        else:
            session_dict['animal'][key] = str

    for key in devices:
        if devices[key] == True:
            session_dict['devices'][key]= True

    session_dict['devices']['implants']['implant_id'] = implant['implant_id']
    session_dict['devices']['implants']['implant_type'] = implant['type']
    session_dict['devices']['implants']['implannt_geometry'] = implant['geometry']
    session_dict['devices']['implants']['implant_data'] = {}   

    return session_dict

def _format_session(cut_file, tet_file, ch_count=4):
    with open(cut_file, 'r') as cut_file1, open(tet_file, 'rb') as tetrode_file1:
        cut_data = _read_cut(cut_file1)
        tetrode_data = _format_spikes(tetrode_file1)
        tetrode_header_data = _read_tetrode_header(tet_file)

    implant_data_dict = _init_implant_data(ch_count)

    implant_data_dict = _fill_implant_data(implant_data_dict, tetrode_data, cut_data, tetrode_header_data, ch_count)

    return implant_data_dict

def _fill_implant_data(implant_data_dict, tetrode_data, cut_data, tetrode_header_data, ch_count):
    implant_data_dict['duration'], implant_data_dict['sample_rate'] = tetrode_header_data['duration'], tetrode_header_data['sample_rate']

    for ch in range(ch_count):
        implant_data_dict['channel_'+str(ch)] = tetrode_data[ch+1]

    implant_data_dict['event_times'] = tetrode_data[0]
    implant_data_dict['event_labels'] = cut_data

    return implant_data_dict

def _init_implant_data(ch_count):
    implant_data_dict = {}

    ch_keys = []
    for ch in range(ch_count):
        key = 'channel_' + str(ch)
        ch_keys.append(key)

    keys = ['event_times', 'event_labels', 'units', 'sample_rate', 'duration']

    keys = np.hstack((keys, ch_keys))

    for key in keys:
        if key == 'units':
            implant_data_dict[key] = 's'
        if key == 'sample_rate':
            implant_data_dict[key] = float
        if key == 'duration':
            implant_data_dict[key] = float
        else:
            implant_data_dict[key] = []

    return implant_data_dict



