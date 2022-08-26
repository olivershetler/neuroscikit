

__version__ = '1.0' # change this to a function for getting release version in future.



# file type is either

lfp_header = {
    'trial_date': 'Tuesday, 22 Jun 2021',
    'trial_time': '11:12:45',
    'experimenter': 'Gus Rodriguez',
    'comments': '',
    'duration': len(lfp_series)*sample_rate,
    'sw_version': 'neuroscikit version {}'.format(__version__),
    'num_chans': '1',
    'sample_rate': '{} Hz'.format(sample_rate),
    '{}_samples_per_position'.format(file_type.upper()): '',
    'bytes_per_sample': str(bytes_per_sample),
    'num_{}_samples'.format(file_type.upper()): str(len(lfp_series)*sample_rate),
    'data_start': '',
}

def make_eeg_or_egf_header(EphysCollection, session_metadata):
    """
    Make the header for the .eeg or .egf file.
    """

    header = {}
    header['trial_date'] = session_metadata['trial_date']
    header['trial_time'] = session_metadata['trial_time']
    header['experimenter'] = session_metadata['experimenter']
    header['comments'] = session_metadata['comments']
    header['duration'] = len(voltage_series)*sample_rate
    header['sw_version'] = 'neuroscikit version {}'.format(__version__)
    header['num_chans'] = '1'
    header['sample_rate'] = '{} Hz'.format(sample_rate)
    header['{}_samples_per_position'.format(file_type.upper())] = ''
    header['bytes_per_sample'] = str(bytes_per_sample)
    header['num_{}_samples'.format(file_type.upper())] = str(len(voltage_series)*sample_rate)
    header['data_start'] = ''
    return header
