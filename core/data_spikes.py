import os
from re import S
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.core_utils import (
    make_seconds_index_from_rate,
)

class SpikeKeys():
    def __init__(self):
        self.sample_length = [
            'sample_length',
            'length',
            'len',
        ]
        self.sample_rate = [
            'sample_rate',
            'rate',
            'r',
        ]
        self.spikes_binary = [
            'spikes_binary',
            'binary',
            'bina',
        ]
        self.spike_times = [
            'spike_times',
            'times',
            'ts',
            't',
            'sptimes',
            'spks',
            'sps',
        ]
        self.cluster_labels = [
            'cluster_labels',
            'labels',
            'label',
            'cluster_label',
        ]

    def get_spike_train_init_keys(self):
        init_keys = [
            'sample_length', 
            'sample_rate', 
            'spikes_binary', 
            'spike_times',
        ]
        return init_keys


class SpikeTypes():
    def __init__(self):
        self.sample_length = int
        self.sample_rate = float
        self.spike_times = list[float]
        self.spikes_binary = list[int]
        self.cluster_labels = list[int]
        self.init_keys = []

    # def _set_init_keys(self, init_keys):
    #     self.init_keys = init_keys

    def type_dict(self):
        types = {
        'sample_length': int(0),
        'sample_rate': float(0.0),
        'spike_times': [],
        'spikes_binary': [],
        'cluster_labels': [],
        }
        return types

    def format_keys(self, init_keys):
        self.init_keys = init_keys

        input_dict = {}
        types = self.type_dict()

        assert len(self.init_keys) != 0, 'make sure to set initial dict. keys using SpikeKeys class'

        for i in range(len(self.init_keys)):
            if self.init_keys[i] in types:
                input_dict[str(self.init_keys[i])] = types[self.init_keys[i]]
        
        return input_dict



class SpikeTrain(): 
    """Single spike train

    Inputs are sample length and sample rate + one of binary spikes or spike times
    """
    spikes_binary: list[int]
    _spike_times: list[float]

    def __init__(self,  input_dict):
        self._input_dict = input_dict
        sample_length, sample_rate, spikes_binary, spike_times = self._read_input_dict()

        self.timestamps = make_seconds_index_from_rate(sample_length, sample_rate)

        assert ((len(spikes_binary) == 0) and (len(spike_times) == 0)) != True, "No spike data provided"

        self._spike_times = spike_times
        self._spikes_binary = spikes_binary
        self._spike_ids = []
        self._spike_rate = None

    def __len__(self):
        if len(self._spike_ids) == 0:
            self.get_binary()
            return len(self._spike_ids)
        else:
            return len(self._spike_ids)

    def __getitem__(self, key):
        return self._spike_ids[key], self._spike_times[key], self._spikes_binary[key]

    def __iter__(self):
        for i in range(len(self._spike_ids)):
            yield self._spike_ids[i], self._spike_times[i], self._spikes_binary[i]

    def __repr__(self):
        return f'SpikeTrain(_spike_ids={self._spike_ids}, _spike_times={self._spike_times}, spikes_binary={self._spikes_binary})'

    def __str__(self):
        return f'SpikeTrain(_spike_ids={self._spike_ids}, _spike_times={self._spike_times}, spikes_binary={self.spikes_binary})'

    def __eq__(self, other):
        return self._spike_ids == other._spike_ids and self._spike_times == other._spike_times and self.spikes_binary == other.spikes_binary

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self._spike_ids, self._spike_times, self.spikes_binary))

    def __lt__(self, other):
        return self._spike_times < other._spike_times

    def __le__(self, other):
        return self._spike_times <= other._spike_times

    def __gt__(self, other):
        return self._spike_times > other._spike_times

    def __ge__(self, other):
        return self._spike_times >= other._spike_times

    def _read_input_dict(self):
        true_keys = list(self._input_dict.keys())
        possible_keys = SpikeKeys()
        spike_data_present = False
        for key in true_keys:
            if key in possible_keys.sample_length:
                sample_length = self._input_dict[key]
            if key in possible_keys.sample_rate:
                sample_rate = self._input_dict[key]
            if key in possible_keys.spikes_binary:
                spikes_binary = self._input_dict[key]
                if len(spikes_binary) > 0:
                    spike_data_present = True
            if key in possible_keys.spike_times:
                spike_times = self._input_dict[key]
                if len(spike_times) > 0:
                    spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        return sample_length, sample_rate, spikes_binary, spike_times

    def _set_spike_rate(self):
        T = (self.timestamps[-1] - self.timestamps[0])
        if len(self._spikes_binary) > 0:
            self._spike_rate = float(sum(self._spikes_binary) / T)
        elif len(self._spike_times) > 0:
             self._spike_rate = float(len(self._spike_times) / T)
        if len(self._spikes_binary) > 0 and len(self._spike_times) > 0:
            assert self._spike_rate == float(len(self._spike_times) / T), 'binary vs timestamps returning different spike rate'

    def get_spike_rate(self):
        if self._spike_rate == None:
            self._set_spike_rate()
            return self._spike_rate
        else:
            return self._spike_rate

    def _set_binary(self):
        for i in range(len(self.timestamps)):
            if self.timestamps[i] in self._spike_times:
                self._spike_ids.append(i)
                self._spikes_binary.append(1)
            else:
                self._spikes_binary.append(0)
        
    def get_binary(self):
        if len(self._spikes_binary) == 0:
            self._set_binary()
            return self._spikes_binary
        else:
            return self._spikes_binary

    def get_spike_times(self):
        if len(self._spike_times) == 0:
            self._set_spike_times()
            return self._spike_times
        else:
            return self._spike_times

    def _set_spike_times(self):
        for i in range(len(self.timestamps)):
            if self._spikes_binary[i] == 1:
                self._spike_ids.append(i)
                self._spike_times.append(self.timestamps[i])




class Spike(): # spike object, has waveforms
    def __init__(self, sample_length, sample_rate, cluster_label, spike_time, ch1, ch2, ch3, ch4):
        assert type(cluster_label) == int, 'Cluster label must be integer for index into waveforms'
        assert type(spike_time) != float, 'Spike times is in format: ' + str(type(spike_time))

        self.timestamps = test_make_seconds_index_from_rate(sample_length, sample_rate)
        self.spike_time = spike_time
        self.label = cluster_label
        self.waveforms = ch1, ch2, ch3, ch4
        # self.waveforms = [self._ch1, self._ch2, self._ch3, self._ch4]
        self._sample_length = sample_length
        self._sample_rate = sample_rate
        self._main_channel = 0
        self._main_waveform = []

    def get_main_channel(self):
        if self._main_channel == 0:
            self._main_channel, self._main_waveform = self._set_main_channel()
            return self._main_channel, self._main_waveform
        else:
            return self._main_channel, self._main_waveform

    def _set_main_channel(self):
        curr = 0
        for i in range(len(self.waveforms)):
            if max(abs(self.waveforms[i])) > curr:
                curr = i + 1
        assert curr != 0, 'There is no 0 channel, make sure max(abs(channel waveform)) is not 0'
        return curr, self.waveforms[curr]


class SpikeClusterBatch():
    # collection of spike cluster objects
    def __init__(self):
        pass

class SpikeCluster(): # collection of spike objects
    """Single spike train

    Inputs are sample length and sample rate + cluster labels and spike times, sorted_cells optional
    """
    def __init__(self, sample_length, sample_rate, cluster_labels, spike_times, sorted_cells=[]):
        assert len(cluster_labels) > 0, 'Cluster labels missing'
        assert len(spike_times) > 0, 'Spike times missing'

        self.timestamps = test_make_seconds_index_from_rate(sample_length, sample_rate)
        self._spike_times = spike_times
        self._labels = cluster_labels
        self._sorted_cells = sorted_cells
        self._sample_length = sample_length
        self._sample_rate = sample_rate
        self._spike_objects = []

    # def _sort_by_label(self):
    #     rng = max(self._labels) + 1 
    #     sorted = [[] for i in range(rng)]
    #     for i in range(len(self._spike_times)):
    #         if self._spike_times[i] == self._labels[i]:
    #             sorted[i].append(self._spike_times[i])
    #     assert len(sorted[-1]) > 0, 'Wrong cell count'
    #     return sorted

    # def _make_spike_train(self, spike_train):
    #     return SpikeTrain(self._sample_length, self._sample_rate, spike_times=spike_train)

    # def get_spike_train_instances(self):
    #     if len(self._sorted_cells) == 0:
    #         sorted = self._sort_by_label()
    #         for i in range(len(sorted)):
    #             spike_train = self._make_spike_train(sorted[i])
    #             self._sorted_cells.append(spike_train)
    #         return self._sorted_cells
    #     else:
    #         return self._sorted_cells

    # def get_binary(self):
    #     self._sorted_cells = self.get_spike_train_instances()
    #     all_binary = []
    #     for spike_train in self._sorted_cells:
    #         all_binary.append(spike_train.get_binary())
    #     return all_binary

    # def get_spike_times(self):
    #     self._sorted_cells = self.get_spike_train_instances()
    #     all_times = []
    #     for spike_train in self._sorted_cells:
    #         all_times.append(spike_train.get_spike_times())
    #     return all_times

    def get_spike_object_instances(self):
        if len(self._spike_objects) == 0:
            for i in range(len(self._spike_times)):
                spike_obj = self._make_spike_object(self)
        pass

    def _make_spike_object(self):
        pass

class SpikeTrainBatch():
    def __init__(self, input_dict):
        self._input_dict = input_dict
        
        sample_length, sample_rate, spikes_binary, spike_times = self._read_input_dict()

        self.timestamps = make_seconds_index_from_rate(sample_length, sample_rate)

        assert ((len(spikes_binary) == 0) and (len(spike_times) == 0)) != True, "No spike data provided"

        self.units = max(len(spikes_binary), len(spike_times))

        self._sample_rate = sample_rate
        self._sample_length = sample_length
        self._spike_times = spike_times
        self._spikes_binary = spikes_binary
        self._spike_ids = []
        self._spike_rate = None
        self._spike_train_instances = []

    def _read_input_dict(self):
        true_keys = list(self._input_dict.keys())
        possible_keys = SpikeKeys()
        spike_data_present = False
        for key in true_keys:
            if key in possible_keys.sample_length:
                sample_length = self._input_dict[key]
            if key in possible_keys.sample_rate:
                sample_rate = self._input_dict[key]
            if key in possible_keys.spikes_binary:
                spikes_binary = self._input_dict[key]
                assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
                if len(spikes_binary) > 0:
                    assert type(spikes_binary[0]) == list, 'Binary spikes are not nested lists (2D), check inputs are not 1D'
                    spike_data_present = True
            if key in possible_keys.spike_times:
                spike_times = self._input_dict[key]
                assert type(spike_times) == list, 'Spike times are not a list, check inputs'
                if len(spike_times) > 0:
                    assert type(spike_times[0]) == list, 'Spike times are not nested lists (2D), check inputs are not 1D'
                    spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        return sample_length, sample_rate, spikes_binary, spike_times

    def _make_spike_train_inputs(self):
        # check was was papssed in, spiek times or binary
        # if len(self._spike_times) > 0:
        #     spike_trains = self._spike_times
        # else:
        #     spike_trains = self._spikes_binary

        # utils classes for filling/checking input dict
        spike_keys = SpikeKeys()
        spike_types = SpikeTypes()
        init_keys = spike_keys.get_spike_train_init_keys()
        # spike_types._set_init_keys(init_keys)

        # arr to collect SpikeTrain() instances
        instances = []

        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in range(self.units):
            input_dict = spike_types.format_keys(init_keys)
            input_dict['sample_length'] = self._sample_length
            input_dict['sample_rate'] = self._sample_rate
            if len(self._spikes_binary) > 0:
                input_dict['spikes_binary'] = self._spikes_binary[i]
            if len(self._spike_times) > 0:
                input_dict['spike_times'] = self._spike_times[i]
            instances.append(self._make_spike_train_instance(input_dict))

        self._spike_train_instances = instances

    def get_spike_train_instances(self):
        if len(self._spike_train_instances) == 0:
            self._make_spike_train_inputs()
        return self._spike_train_instances

    def _make_spike_train_instance(self, input_dict):
        return SpikeTrain(input_dict)

    def get_indiv_spike_rate(self):
        self.get_spike_train_instances()
        spike_rates = []
        for i in range(len(self._spike_train_instances)):
            spike_rates.append(self._spike_train_instances[i].get_spike_rate())
        return spike_rates

    def get_average_spike_rate(self):
        self.get_spike_train_instances()
        spike_rates = self.get_indiv_spike_rate()
        return sum(spike_rates) / len(spike_rates)

    def _set_binary(self):
        for i in range(len(self._spike_train_instances)):
            self._spikes_binary.append(self._spike_train_instances[i].get_binary())
        return self._spikes_binary

    def _set_spike_times(self):
        for spike_train in self._spike_train_instances:
            self._spike_times.append(spike_train.get_spike_times())
        return self._spike_times
        
    # get 2d spike inputs as binary
    def get_binary(self):
        if len(self._spikes_binary) == 0:
            self.get_spike_train_instances()
            self._set_binary()
        else:
            return self._spikes_binary

    # get 2d spike inputs as timestamps
    def get_spike_times(self):
        if len(self._spike_times) == 0:
            self.get_spike_train_instances()
            self._set_spike_times()
        else:
            return self._spike_times

    



    # check if the spike train is in time-stamp format or binary format
    # if in binary format, create binary data object and extract timestamp format as well
    # if in time-stamp format, create time-stamp data object onlys
    # have method for storing unbinned and binned rate estimatess

# class TimeStampSpikeTrain(SpikeTrain):
#     """ Spike train as time stamps 
#     """

#     def __init(self, _spike_times, spike_features):
#         self._spike_times = _spike_times

# class BinarySpikeTrain(SpikeTrain):
#     """ Spike train as binary 
#     """

#     def __init(self, spikes_binary, spike_features):
#         self.spikes_binary = spikes_binary


