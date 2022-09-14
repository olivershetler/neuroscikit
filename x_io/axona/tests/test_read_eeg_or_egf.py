import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

from x_io.axona.read_eeg_or_egf import (
    read_eeg_or_egf
    ,_make_lfp_object
    ,load_eeg_or_egf_from_path
)

import os


cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_eeg_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90.eeg')
#test_egf_file_path = os.path.join(data_dir, 'axona/20140815-behavior2-90.egf')

def test_read_eeg_or_egf():
    with open(test_eeg_file_path, 'rb') as eeg_file:
        eeg_values = read_eeg_or_egf(eeg_file, 'eeg')
    print(type(eeg_values))

def test__make_lfp_object():
    with open(test_eeg_file_path, 'rb') as eeg_file:
        eeg_values = read_eeg_or_egf(eeg_file, 'eeg')
    eeg = _make_lfp_object(eeg_values, 250)
    print(type(eeg))


