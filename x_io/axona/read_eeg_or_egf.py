"""
This module contains methods for reading and writing data to the .eeg and .egf file formats, which store local field potential (LFP) data.
"""

# Internal Dependencies

# A+ Grade Dependencies

# A Grade Dependencies

# Other Dependencies


def read_eeg_or_egf(eeg_file, file_type='eeg'):
    """input:
    eeg_filename: the fullpath to the eeg file that is desired to be read.
    Example: C:\Location\of\eegfile.eegX

    Output:
    The EEG waveform, and the sampling frequency
    """

    with open(eeg_fname, 'rb') as f:

        is_eeg = False
        is_egf = False
        if 'eeg' in file_type:
            is_eeg = True
        if 'egf' in file_type:
            is_egf = True
        else:
            raise ValueError('The file extension must be either "eeg" or "egf"')


        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            # find the data_start
            start_index = int(m.find(b'data_start') + len('data_start'))  # start of the data
            stop_index = int(m.find(b'\r\ndata_end'))  # end of the data

            sample_rate_start = m.find(b'sample_rate')
            sample_rate_end = m[sample_rate_start:].find(b'\r\n')
            Fs = float(m[sample_rate_start:sample_rate_start + sample_rate_end].decode('utf-8').split(' ')[1])

            m = m[start_index:stop_index]

            if is_eeg:
                assert Fs == 250
                EEG = np.fromstring(m, dtype='>b')
            else:
                assert Fs == 4.8e3
                EEG = np.fromstring(m, dtype='<h')

            return EEG