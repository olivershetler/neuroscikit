import os
import sys
import numpy as np

prototype_dir = os.getcwd()
sys.path.append(prototype_dir)
parent_dir = os.path.dirname(prototype_dir)

from _prototypes.cell_remapping.main import batch_remapping

# data_dir = parent_dir + r'\neuroscikit_test_data\RadhaData\Data\lowPHF'
# data_dir = parent_dir + r'\neuroscikit_test_data\20180502-ROUND-3000_ONLY_TET_2'
# data_dir = parent_dir + r'\neuroscikit_test_data\20180502-ROUND-3000'
data_dir = parent_dir + r'\neuroscikit_test_data\Test_File_Andrew-20221216T163437Z-001'

animal = {'animal_id': 'id', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
devices = {'axona_led_tracker': True, 'implant': True}
implant = {'implant_id': 'id', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

settings_dict = {'ppm': 511, 'session': session_settings, 'smoothing_factor': 3, 'useMatchedCut': True, 'hasObject': True}

def test_batch_remapping():
    batch_remapping([data_dir], settings_dict)


if __name__ == '__main__':
    test_batch_remapping()
