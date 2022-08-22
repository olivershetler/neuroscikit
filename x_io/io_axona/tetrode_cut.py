"""
This module contains methods for reading and writing to the .X (tetrode) and .cut file formats, which store extracted spikes and spike cluster data.
"""

# Internal Dependencies
from core.data_spikes import SpikeTrain

# A+ Grade Dependencies
import numpy as np
import os

# A Grade Dependencies
from __future__ import division, print_function
import numpy.matlib as mlb

# Other Dependencies
# None

def _read_cut(cut_filename):
    """This function will read the given cut file, and output the """
    cut_values = None
    if os.path.exists(cut_filename):
        extract_cut = False
        with open(cut_filename, 'r') as f:
            for line in f:
                if 'Exact_cut' in line:  # finding the beginning of the cut values
                    extract_cut = True
                if extract_cut:  # read all the cut values
                    cut_values = str(f.readlines())
                    for string_val in ['\\n', ',', "'", '[', ']']:  # removing non base10 integer values
                        cut_values = cut_values.replace(string_val, '')
                    cut_values = [int(val) for val in cut_values.split()]
        cut_values = np.asarray(cut_values)
    return cut_values

''' this function seems not to be called anywhere
def _find_unit(tetrode_list):
    """Inputs:
    tetrode_list: list of tetrodes to find the units that are in the tetrode_path
    example [r'C:Location\of\File\filename.1', r'C:Location\of\File\filename.2' ],
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

        cut_filename = os.path.join(directory, '%s_%d.cut' % (tetrode_base, tetrode))

        cut_values = _read_cut(cut_filename)
        cut_list.append(cut_values)
        unique_cell_list.append(np.unique(cut_values))

    return cut_list
'''

def _importspikes(filename):
    """Reads through the tetrode file as an input and returns two things, a dictionary containing the following:
    timestamps, ch1-ch4 waveforms, and it also returns a dictionary containing the spike parameters"""

    with open(filename, 'rb') as f:
        for line in f:
            if 'data_start' in str(line):
                spike_data = np.fromstring((line + f.read())[len('data_start'):-len('\r\ndata_end\r\n')], dtype='uint8')
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

                # calculating the big-endian and little endian matrices so we can convert from bytes -> decimal

    big_endian_vector = 256 ** np.arange(bytes_per_timestamp - 1, -1, -1)
    little_endian_matrix = np.arange(0, bytes_per_sample).reshape(bytes_per_sample, 1)
    little_endian_matrix = 256 ** mlb.repmat(little_endian_matrix, 1, samples_per_spike)

    number_channels = 4

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

    return {'t': t.reshape(num_spikes, 1), 'ch1': np.asarray(waveform_data[0][:][:]),
            'ch2': np.asarray(waveform_data[1][:][:]),
            'ch3': np.asarray(waveform_data[2][:][:]), 'ch4': np.asarray(waveform_data[3][:][:])}, spikeparam

def _getspikes(fullpath):
    """
    This function will return the spike data, spike times, and spike parameters from Tint tetrode data.

    Example:
        tetrode_fullpath = 'C:\\example\\tetrode_1.1'
        ts, ch1, ch2, ch3, ch4, spikeparam = _getspikes(tetrode_fullpath)

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
    spikes, spikeparam = _importspikes(fullpath)
    ts = spikes['t']
    nspk = spikeparam['num_spikes']
    spikelen = spikeparam['samples_per_spike']

    ch1 = spikes['ch1']
    ch2 = spikes['ch2']
    ch3 = spikes['ch3']
    ch4 = spikes['ch4']

    return ts, ch1, ch2, ch3, ch4, spikeparam

#called in batch_processing, compute_all_map_data, npy_converter and shuffle_processing
def load_neurons(cut_path: str, tetrode_path: str, channel_no: int) -> tuple:

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
    '''

    # Read cut and tetrode data
    cut_data = _read_cut(cut_path)
    tetrode_data = _getspikes(tetrode_path)
    number_of_neurons = max(cut_data) + 1

    # Organize neuron data into list
    channel = [[[] for x in range(2)]for x in range(number_of_neurons)]

    for i in range(len(tetrode_data[0])):
        channel[ cut_data[i] ][0].append(tetrode_data[channel_no][i])
        channel[ cut_data[i] ][1].append(float(tetrode_data[0][i]))

    # Find where there is a break in the neuron data
    # and assign the empty space number as the empty cell

    for i, element in enumerate(channel):
        if (len(element[0]) == 0 or len(element[1]) == 0) and i != 0:
            empty_cell = i
            break
        else:
            empty_cell = i + 1


    return channel, empty_cell