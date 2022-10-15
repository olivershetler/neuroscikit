import os
import sys


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.instruments import TrackerMetadata, DevicesMetadata, ImplantMetadata

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data/test_dir')
cut_file = os.path.join(data_dir, '20140815-behavior2-90_1.cut')
tet_file = os.path.join(data_dir, '20140815-behavior2-90.1')
pos_file = os.path.join(data_dir, '20140815-behavior2-90.pos')

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
pos_dict = {'x' : [0,1,2,3], 'y': [0,1,2,3], 'arena_height': 1, 'arena_width': 1, 't': [0,1,2,3]}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}


settings_dict = {'ppm': 511, 'session':  session_settings}


def test_session_tracker():
    session_tracker = TrackerMetadata(pos_dict)

    assert isinstance(session_tracker, TrackerMetadata)
    assert session_tracker.arena_height != None
    if type(session_tracker.y) == list:
        assert session_tracker.y != None
        assert session_tracker.x != None
        assert session_tracker.time != None
    else:
        assert session_tracker.y.all() != None
        assert session_tracker.x.all() != None
        assert session_tracker.time.all() != None
    assert session_tracker.arena_width != None

def test_session_implant():
    session_implant = ImplantMetadata(implant)

    assert isinstance(session_implant, ImplantMetadata)
    assert session_implant.implant_id == implant['implant_id']
    assert session_implant.implant_type == implant['implant_type']
    assert session_implant.implant_units == implant['implant_units']

def test_session_devices():
    session_devices = DevicesMetadata({'implant': ImplantMetadata(implant), 'axona_led_tracker': TrackerMetadata(pos_dict)})

    assert isinstance(session_devices, DevicesMetadata)
    assert type(session_devices.devices_dict) == dict
    assert isinstance(session_devices.devices_dict['axona_led_tracker'], TrackerMetadata)
    assert isinstance(session_devices.devices_dict['implant'], ImplantMetadata)
