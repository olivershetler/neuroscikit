import os
import struct

__version__ = '1.0' # change this to a function for getting release version in future.


def write_eeg_or_egf(EphysCollection, session_metadata, target_directory, session_name):
    data = EphysCollection.data
    assert len(EphysCollection.data) > 0, "EphysCollection is empty"
    for i, channel_name in enumerate(data.keys()):
        sample_rate = data[channel_name].sample_rate
        if sample_rate[0] <= 500:
            file_type = 'eeg'
        elif sample_rate[0] > 500:
            file_type = 'egf'


        if i > 0:
            n = str(i)
        else:
            n = ''
        path = os.path.join(target_directory, '{}.{}{}'.format(session_name, file_type, n))

        header = make_eeg_or_egf_header(EphysCollection.data[channel_name], session_metadata)

        with open(path, 'w') as f:
            for key, value in header.items():
                f.write('{} {}\n'.format(key, value))
            f.write('data_start')
        series = data[channel_name].data
        num_samples = len(series)
        if file_type == 'eeg':
            binary_data = struct.pack('>%db' % (
            ), *[np.int(data_value) for data_value in series.tolist()])
        elif file_type == 'egf':
            binary_data = struct.pack('<%dh' % (num_samples), *[int(data_value) for data_value in series.tolist()])
        with open(path, 'rb+') as f:
            f.seek(0, 2)
            f.write(binary_data)
            f.write(bytes('\r\ndata_end\r\n', 'utf-8'))



def make_eeg_or_egf_header(EphysSeries, session_metadata):
    """
    Make the header for the .eeg or .egf file.
    """

    voltage_series = EphysSeries.data
    sample_rate = EphysSeries.sample_rate

    if sample_rate[0] <= 500:
        #sample_rate is of the form (rate, 'Hz').
        file_type = 'eeg'
        bytes_per_sample = 1
        num_samples = len(voltage_series)
    elif sample_rate[0] > 500:
        file_type = 'egf'
        bytes_per_sample = 2
        num_samples = len(voltage_series)

    header = {}
    header['trial_date'] = session_metadata['trial_date']
    header['trial_time'] = session_metadata['trial_time']
    header['experimenter'] = session_metadata['experimenter']
    header['comments'] = session_metadata['comments']
    header['duration'] = len(voltage_series)*sample_rate[0]
    header['sw_version'] = 'neuroscikit version {}'.format(__version__)
    header['num_chans'] = '1'
    header['sample_rate'] = '{} {}'.format(sample_rate[0], sample_rate[1])
    header['bytes_per_sample'] = str(bytes_per_sample)
    header['num_{}_samples'.format(file_type.upper())] = str(len(voltage_series))

    return header
