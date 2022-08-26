"""
This module contains methods for reading and writing data to the .eeg and .egf file formats, which store local field potential (LFP) data.
"""


# Internal Dependencies
import contextlib # for closing the file
import mmap

# A+ Grade Dependencies
import numpy as np
import _io

# A Grade Dependencies

# Other Dependencies


def read_eeg_or_egf(opened_eeg_or_egf_file: _io.BufferedReader, file_type: str) -> np.ndarray:
    """input:
    opened_eeg_or_egf_file: an open file object for the .eeg or .egf file
    Output:
    The EEG waveform, and the sampling frequency
    """
    is_eeg = False
    is_egf = False
    if 'eeg' == file_type:
        is_eeg = True
    elif 'egf' == file_type:
        is_egf = True
    else:
        raise ValueError('The file extension must be either "eeg" or "egf".')

    assert is_eeg != is_egf, 'The file extension must be _either_ "eeg" _xor_ "egf". Currently the is_eeg = {}, and is_egf = {}'.format(is_eeg, is_egf)


    with contextlib.closing(mmap.mmap(opened_eeg_or_egf_file.fileno(), 0, access=mmap.ACCESS_READ)) as memory_map:
        # find the data_start
        start_index = int(memory_map.find(b'data_start') + len('data_start'))  # start of the data
        stop_index = int(memory_map.find(b'\r\ndata_end'))  # end of the data

        sample_rate_start = memory_map.find(b'sample_rate')
        sample_rate_end = memory_map[sample_rate_start:].find(b'\r\n')
        Fs = float(memory_map[sample_rate_start:sample_rate_start + sample_rate_end].decode('utf-8').split(' ')[1])

        data_string = memory_map[start_index:stop_index]

        if is_eeg and not is_egf:
            assert Fs == 250
            eeg = np.frombuffer(data_string, dtype='>b')
            return eeg
        elif is_egf and not is_eeg:
            assert Fs == 4.8e3
            egf = np.frombuffer(data_string, dtype='<h')
            return egf
        else:
            raise ValueError('The file extension must be either "eeg" or "egf"')

# ----------------------------------------------------------------------------- #
# Wrappers for the above functions when the
# #data are stored as files in folders in one
# of the three major OSes (Windows, Mac, Linux).

def load_eeg_or_egf_from_path(file_path):
    with open(file_path, 'rb') as eeg_or_egf_file:
        if '.eeg' in file_path:
            return read_eeg_or_egf(eeg_or_egf_file, file_type='eeg')
        elif '.egf' in file_path:
            return read_eeg_or_egf(eeg_or_egf_file, file_type='egf')
        else:
            raise ValueError('The file extension must be either "eeg" or "egf". The current file extension is .{}'.format(file_path.split('.')[-1]))