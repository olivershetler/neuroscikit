
#TODO: set up gin ssh key for https://gin.g-node.org/

#TODO: load test data to the gin repository

#TODO: figure out how to read the data directly from the internet.

# eventually replace this with urllib
# access to the data online

from csv import DictReader
import os
import sys
from turtle import pos


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

 
from x_io.rw.axona.read_pos import grab_position_data

from x_io.rw.axona.batch_read import (
    make_session,
    make_study,
)

from x_io.session import SessionAnimal, SessionDevices, SessionImplant, SessionTracker

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data/test_dir')
cut_file = os.path.join(data_dir, '20140815-behavior2-90_1.cut')
tet_file = os.path.join(data_dir, '20140815-behavior2-90.1')
pos_file = os.path.join(data_dir, '20140815-behavior2-90.pos')

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}


settings_dict = {'ppm': 511, 'sessions': [session_settings,]}

def test_session_animal():
    session_animal = SessionAnimal(animal)

    assert isinstance(session_animal, SessionAnimal)
    assert session_animal.animal_id == animal['animal_id']
    assert session_animal.species == animal['species']
    assert session_animal.weight == animal['weight']

def test_session_tracker():
    pos_dict = grab_position_data(pos_file, settings_dict['ppm'])
    session_tracker = SessionTracker(pos_dict)

    assert isinstance(session_tracker, SessionTracker)
    assert session_tracker.arena_height != None
    assert session_tracker.y.all() != None
    assert session_tracker.x.all() != None
    assert session_tracker.time.all() != None
    assert session_tracker.arena_width != None

def test_session_implant():
    session_implant = SessionImplant(implant)

    assert isinstance(session_implant, SessionImplant)
    assert session_implant.implant_id == implant['implant_id']
    assert session_implant.implant_type == implant['implant_type']
    assert session_implant.implant_units == implant['implant_units']

def test_session_devices():
    pos_dict = grab_position_data(pos_file, settings_dict['ppm'])
    session_devices = SessionDevices({'implant': implant, 'axona_led_tracker': pos_dict})

    assert isinstance(session_devices, SessionDevices)
    assert type(session_devices.devices_dict) == dict
    assert isinstance(session_devices.devices_dict['axona_led_tracker'], SessionTracker)
    assert isinstance(session_devices.devices_dict['implant'], SessionImplant)

def test_session():
    session = make_session(cut_file, tet_file, pos_file, session_settings, settings_dict['ppm'])

    assert isinstance(session.animal, SessionAnimal)
    assert isinstance(session.devices, SessionDevices)
    assert isinstance(session.devices.devices_dict['axona_led_tracker'], SessionTracker)
    assert isinstance(session.devices.devices_dict['implant'], SessionImplant)


    