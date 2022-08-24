"""
This module contains methods for reading and writing to the .X (tetrode) and .cut file formats, which store extracted spikes and spike cluster data.

We get spike trains from these two files combined.
"""
from __future__ import division, print_function

# Internal Dependencies
#from core.data_spikes import SpikeTrain
from x_io.os_io_utils import with_open_file

# A+ Grade Dependencies
import numpy as np
import os

# A Grade Dependencies
# None

# Other Dependencies
# None

def _read_cut(cut_file):
    """This function takes a pre-opened cut file and returns a list of the neuron numbers in the file."""
    cut_values = None
    extract_cut = False
    for line in cut_file:
        if 'Exact_cut' in line:  # finding the beginning of the cut values
            extract_cut = True
        if extract_cut:  # read all the cut values
            cut_values = str(cut_file.readlines())
            for string_val in ['\\n', ',', "'", '[', ']']:  # removing non base10 integer values
                cut_values = cut_values.replace(string_val, '')
            cut_values = [int(val) for val in cut_values.split()]
        else:
            continue
    if cut_values is None:
        raise ValueError('Either file does not exist or there are no cut values found in cut file.')
        cut_values = np.asarray(cut_values)
    return cut_values

''' this function seems not to be called anywhere
def _find_unit(tetrode_list):
    """Inputs:
    tetrode_list: list of tetrodes to find the units that are in the tetrode_path
    example [r'C:Location/of/File/filename.1', r'C:Location/of/File/filename.2' ],
    -------------------------------------------------------------
    Outputs:
    cut_list: an nx1 list for n-tetrodes in the tetrode_list containing a list of unit numbers that each spike belongs to
    """

    input_list = True
    if type(tetrode_list) != list:
        input_list = False
        tetrode_list = [tetrode_list]

    cut_list = []
    unique_cell_list = []
    for tetrode_file in tetrode_list:
        directory = os.path.dirname(tetrode_file)

        try:
            tetrode = int(os.path.splitext(tetrode_file)[1][1:])
        except ValueError:
            raise ValueError("The following file is invalid: %s" % tetrode_file)

        tetrode_base = os.path.splitext(os.path.basename(tetrode_file))[0]

        cut_file_path = os.path.join(directory, '%s_%d.cut' % (tetrode_base, tetrode))

        cut_values = _read_cut(cut_file_path)
        cut_list.append(cut_values)
        unique_cell_list.append(np.unique(cut_values))

    return cut_list
'''

def _read_tetrode_header(tetrode_file):
    """
    This function will return the header values from the Tint tetrode file.
    Example:
        tetrode_fullpath = 'C:\\example\\tetrode_1.1'
        spikeparam = _read_tetrode_header(tetrode_fullpath)

    Args:
        pre-opened file: the fullpath to the Tint tetrode file you want to acquire the spike data from.

    Returns:
        spikeparam (dict): a dictionary containing the header values from the tetrode file.

    Example header lines:
        trial_date Friday, 15 Aug 2014
        trial_time 15:21:10
        experimenter Abid
        comments
        duration 1801
        sw_version 1.2.2.16
        num_chans 4
        timebase 96000 hz
        bytes_per_timestamp 4
        samples_per_spike 50
        sample_rate 48000 hz
        bytes_per_sample 1
        spike_format t,ch1,t,ch2,t,ch3,t,ch4
        num_spikes 200492
    """
    int_fields = {'duration', 'num_chans', 'bytes_per_timestamp', 'samples_per_spike', 'bytes_per_sample', 'num_spikes'}
    float_with_unit_fields = {'timebase', 'sample_rate'}
    comma_lists = {'spike_format'}
    header = dict()
    for line in tetrode_file:
        if 'data_start' in str(line):
            return header
        else:
            field, value = line.decode(encoding='UTF-8').split(' ', 1)
            field, value = field.strip(), value.strip()
            if field in float_with_unit_fields:
                numeric, unit = value.split(' ')
                header[field] = (float(numeric), unit)
            elif field in int_fields:
                header[field] = int(value)
            elif field in comma_lists:
                header[field] = value.split(',')
            else:
                header[field] = value
    return header

def _read_tetrode_data(tetrode_file):
    pass


def _read_tetrode(tetrode_file):
    """Reads through the tetrode file as an input and returns two things, a dictionary containing the following:
    timestamps, ch1-ch4 waveforms, and it also returns a dictionary containing the spike parameters"""

    for line in tetrode_file:
        if 'data_start' in str(line):
            data_string = (line + tetrode_file.read())[len('data_start'):-len('\r\ndata_end\r\n')]
            spike_data = np.frombuffer(data_string, dtype='uint8')
            break
        elif 'num_spikes' in str(line):
            num_spikes = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'bytes_per_timestamp' in str(line):
            bytes_per_timestamp = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'samples_per_spike' in str(line):
            samples_per_spike = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'bytes_per_sample' in str(line):
            bytes_per_sample = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'timebase' in str(line):
            timebase = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'duration' in str(line):
            duration = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'sample_rate' in str(line):
            samp_rate = int(line.decode(encoding='UTF-8').split(" ")[1])
        elif 'num_chans' in str(line):
            num_chans = int(line.decode(encoding='UTF-8').split(" ")[1])

    # calculating the big-endian and little endian matrices so we can convert from bytes -> decimal

    big_endian_vector = 256 ** np.arange(bytes_per_timestamp - 1, -1, -1)
    little_endian_matrix = np.arange(0, bytes_per_sample).reshape(bytes_per_sample, 1)
    little_endian_matrix = 256 ** np.tile(little_endian_matrix, (1, samples_per_spike))

    number_channels = num_chans # should this be hardcoded?
    # can it be detected?

    # calculating the timestamps
    t_start_indices = np.linspace(0, num_spikes * (bytes_per_sample * samples_per_spike * 4 +
                                                   bytes_per_timestamp * 4), num=num_spikes, endpoint=False).astype(
        int).reshape(num_spikes, 1)
    t_indices = t_start_indices

    for chan in np.arange(1, number_channels):
        t_indices = np.hstack((t_indices, t_start_indices + chan))

    t = spike_data[t_indices].reshape(num_spikes, bytes_per_timestamp)  # acquiring the time bytes
    t = np.sum(np.multiply(t, big_endian_vector), axis=1) / timebase  # converting from bytes to float values
    t_indices = None

    waveform_data = np.zeros((number_channels, num_spikes, samples_per_spike))  # (dimensions, rows, columns)

    bytes_offset = 0
    # read the t,ch1,t,ch2,t,ch3,t,ch4

    for chan in range(number_channels):  # only really care about the first time that gets written
        chan_start_indices = t_start_indices + chan * samples_per_spike + bytes_per_timestamp + bytes_per_timestamp * chan
        for spike_sample in np.arange(1, samples_per_spike):
            chan_start_indices = np.hstack((chan_start_indices, t_start_indices +
                                            chan * samples_per_spike + bytes_per_timestamp +
                                            bytes_per_timestamp * chan + spike_sample))
        waveform_data[chan][:][:] = spike_data[chan_start_indices].reshape(num_spikes, samples_per_spike).astype(
            'int8')  # acquiring the channel bytes
        waveform_data[chan][:][:][np.where(waveform_data[chan][:][:] > 127)] -= 256
        waveform_data[chan][:][:] = np.multiply(waveform_data[chan][:][:], little_endian_matrix)

    spikeparam = {'timebase': timebase, 'bytes_per_sample': bytes_per_sample, 'samples_per_spike': samples_per_spike,
                  'bytes_per_timestamp': bytes_per_timestamp, 'duration': duration, 'num_spikes': num_spikes,
                  'sample_rate': samp_rate}
    ephys_data = {'t': t.reshape(num_spikes, 1)}
    for chan in range(number_channels):
        ephys_data['ch%d' % (chan + 1)] = np.asarray(waveform_data[chan][:][:])

    return ephys_data, spikeparam


def _format_spikes(tetrode_file):
    """
    This function will return the spike data, spike times, and spike parameters from Tint tetrode data.

    Example:
        tetrode_fullpath = 'C:\\example\\tetrode_1.1'
        ts, ch1, ch2, ch3, ch4, spikeparam = _format_spikes(tetrode_fullpath)

    Args:
        fullpath (str): the fullpath to the Tint tetrode file you want to acquire the spike data from.

    Returns:
        ts (ndarray): an Nx1 array for the spike times, where N is the number of spikes.
        ch1 (ndarray) an NxM matrix containing the spike data for channel 1, N is the number of spikes,
            and M is the chunk length.
        ch2 (ndarray) an NxM matrix containing the spike data for channel 2, N is the number of spikes,
            and M is the chunk length.
        ch3 (ndarray) an NxM matrix containing the spike data for channel 3, N is the number of spikes,
            and M is the chunk length.
        ch4 (ndarray) an NxM matrix containing the spike data for channel 4, N is the number of spikes,
            and M is the chunk length.
        spikeparam (dict): a dictionary containing the header values from the tetrode file.
    """
    spikes, spikeparam = _read_tetrode(tetrode_file)
    ts = spikes['t']
    nspk = spikeparam['num_spikes']
    spikelen = spikeparam['samples_per_spike']

    ch1 = spikes['ch1']
    ch2 = spikes['ch2']
    ch3 = spikes['ch3']
    ch4 = spikes['ch4']

    return ts, ch1, ch2, ch3, ch4, spikeparam

#called in batch_processing, compute_all_map_data, npy_converter and shuffle_processing
def get_spike_trains_from_channel(open_cut_file, open_tetrode_file, channel_no: int) -> tuple:

    '''
        Loads the neuron of interest from a specific cut file.

        Params:
            open_cut_file:
                A pre-opened cut file.
            tetrode_path (str):
                The path of the tetrode file
            channel_no (int):
                The channel number (1,2,3 or 4)

        Returns:
            Tuple: (channel_data, empty_cell_number)
            --------
            channel_data (list):
                Nested list of all the firing data per neuron
            empty_cell_number (int):
                The 'gap' cell which indicates where the program should stop
                reading cells from Tint.
        Assumptions:
            Only the first few cells are good cells. Anything stored after an empty cell is not a good cell (i.e. it's noise or something that can't accurately be converted to a spike train).
    '''

    # Read cut and tetrode data
    cut_data = _read_cut(open_cut_file)
    tetrode_data = _format_spikes(open_tetrode_file)
    number_of_neurons = max(cut_data) + 1

    # Organize neuron data into list
    channel = [[[] for x in range(2)] for x in range(number_of_neurons)]

    for i in range(len(tetrode_data[0])):
        channel[cut_data[i]][0].append(tetrode_data[channel_no][i])
        channel[cut_data[i]][1].append(float(tetrode_data[0][i]))

    # Find where there is a break in the neuron data
    # and assign the empty space number as the empty cell

    for i, element in enumerate(channel):
        if (len(element[0]) == 0 or len(element[1]) == 0) and i != 0:
            empty_cell = i
            break
        else:
            empty_cell = i + 1


    return channel, empty_cell

#-----------------------------------------------------------------------------------------------------------------------#
# OS wrapper function(s)
#-----------------------------------------------------------------------------------------------------------------------#

def load_spike_train_from_paths(cut_path: str, tetrode_path: str, channel_no: int) -> tuple:

    '''
        Loads the neuron of interest from a specific cut file.

        Params:
            cut_path (str):
                The path of the cut file
            tetrode_path (str):
                The path of the tetrode file
            channel_no (int):
                The channel number (1,2,3 or 4)

        Returns:
            Tuple: (channel_data, empty_cell_number)
            --------
            channel_data (list):
                Nested list of all the firing data per neuron
            empty_cell_number (int):
                The 'gap' cell which indicates where the program should stop
                reading cells from Tint.
        Assumptions:
            Only the first few cells are good cells. Anything stored after an empty cell is not a good cell (i.e. it's noise or something that can't accurately be converted to a spike train).
    '''

    # Read cut and tetrode data
    assert os.path.exists(cut_path), "Cut file does not exist"
    assert os.path.exists(tetrode_path), "Tetrode file does not exist"
    assert cut_path[-4:] == ".cut", "Cut file is not a .cut file"
    #extension = tetrode_path.split('.')[-1]
    #assert int(extension) >= 0, "Tetrode file path does not lead to a valid tetrode file. The extension must be a positive integer (n>=0)."
    with open(cut_path, 'r') as cut_file, open(tetrode_path, 'rb') as tetrode_file:
        channel, empty_cell = get_spike_trains_from_channel(cut_file, tetrode_file, 1)

    return channel, empty_cell
