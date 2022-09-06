from enum import unique
import os
from re import S
import sys
import wave

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.core_utils import (
    make_seconds_index_from_rate,
)

class InputKeys():
    """
    Helper class with ordered channel keys and init dicts for class to class instantiation (e.g. SpikeTrainBatch() --> multiple SpikeTrain())
    """
    def __init__(self):
        pass

    def get_spike_train_init_keys(self):
        init_keys = [
            'sample_length', 
            'sample_rate', 
            'spikes_binary', 
            'spike_times',
        ]
        return init_keys

    def get_spike_cluster_init_keys(self):
        init_keys = [
            'sample_length', 
            'sample_rate', 
            'spike_times',
            'cluster_labels',
            'ch1','ch2','ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'
        ]
        return init_keys

    def get_channel_keys(self):
        init_keys = ['ch1','ch2','ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        return init_keys

class SpikeTrain(): 
    """
    Class to hold 1D spike train, spike train is a sorted set of spikes belonging to one cluster
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
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        spikes_binary = self._input_dict['spikes_binary']
        assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
        if len(spikes_binary) > 0:
            spike_data_present = True
        spike_times = self._input_dict['spike_times']
        assert type(spike_times) == list, 'Spike times are not a list, check inputs'
        if len(spike_times) > 0:
            spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        return sample_length, sample_rate, spikes_binary, spike_times

    # lazy eval, called only if get spike rate called and spiek rate not pre filled
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

    # lazy eval, returns binary from spike times only if get binary called
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

    # lazy eval, returns spike times from binary spikes only if get spikes times called
    def _set_spike_times(self):
        for i in range(len(self.timestamps)):
            if self._spikes_binary[i] == 1:
                self._spike_ids.append(i)
                self._spike_times.append(self.timestamps[i])

class Spike(): # spike object, has waveforms
    """
    Class to hold single spike object and waveforms associated with it
    collection of Spike() = SpikeCluster()
    """
    def __init__(self, input_dict):
        self._input_dict = input_dict
        sample_length, sample_rate, cluster_label, spike_time, waveforms = self._read_input_dict()

        assert type(cluster_label) == int, 'Cluster label must be integer for index into waveforms'
        assert type(spike_time) == float, 'Spike times is in format: ' + str(type(spike_time))

        self.timestamps = make_seconds_index_from_rate(sample_length, sample_rate)
        self.spike_time = spike_time
        self.label = cluster_label
        self.waveforms = waveforms
        # self.waveforms = [self._ch1, self._ch2, self._ch3, self._ch4]
        self._sample_length = sample_length
        self._sample_rate = sample_rate
        self._main_channel = 0
        self._main_waveform = []

    # Organize waveforms by channel in ascending order: ch1, ch2, etc...
    def _extract_waveforms(self):
        input_keys = InputKeys()
        channel_keys = input_keys.get_channel_keys()
        waveforms = []
        for i in range(len(channel_keys)):
            if channel_keys[i] in self._input_dict.keys():
                waveforms.append(self._input_dict[channel_keys[i]])
        return waveforms

    def _read_input_dict(self):
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        # spikes_binary = self._input_dict['spikes_binary']
        cluster_label = self._input_dict['cluster_label']
        # assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
        # if len(spikes_binary) > 0:
            # spike_data_present = True
        spike_time = self._input_dict['spike_time']
        assert type(spike_time) == float, 'Spike time must be single number'

        waveforms = self._extract_waveforms()
        return sample_length, sample_rate, cluster_label, spike_time, waveforms

    # one waveform per channel bcs class is for one spike
    def get_single_channel_waveform(self, id):
        assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
        return self.waveforms[id-1]

    # get waveform with largest positive or negative deflection (peak or trough, absolute val)
    def get_main_channel(self):
        if self._main_channel == 0:
            self._main_channel, self._main_waveform = self._set_main_channel()
            return self._main_channel, self._main_waveform
        else:
            return self._main_channel, self._main_waveform

    # lazy eval, called when get main channel called
    def _set_main_channel(self):
        curr = 0
        for i in range(len(self.waveforms)):
            for j in range(len(self.waveforms[i])):
                if abs(self.waveforms[i][j]) > curr:
                    curr = i + 1
        assert curr != 0, 'There is no 0 channel, make sure max(abs(channel waveform)) is not 0'
        return curr, self.waveforms[curr-1]

    # # cluster label for a given spike train
    # def set_cluster_label(self, label):
    #     self.label = label
    
    # def get_cluster_label(self):
    #     return self.label

class SpikeClusterBatch():
    """
    Class to batch process SpikeClusters. Can pass in unorganized set of 1D spike times + cluster labels
    will create a collection of spike clusters with a collection of spikie objects in each cluster
    """
    def __init__(self, input_dict):
        self._input_dict = input_dict
        sample_length, sample_rate, cluster_labels, spike_times, waveforms = self._read_input_dict()

        assert type(cluster_labels) == list, 'Cluster labels missing'
        assert type(cluster_labels[0]) == int, 'Cluster labels missing'
        assert len(spike_times) > 0, 'Spike times missing'
        assert len(waveforms) <= 8 and len(waveforms) > 0, 'Cannot have fewer than 0 or more than 8 channels'

        self.timestamps = make_seconds_index_from_rate(sample_length, sample_rate)

        self._spike_times = spike_times
        self._cluster_labels = cluster_labels
        self._sample_length = sample_length
        self._sample_rate = sample_rate
        self._spike_clusters = []
        self._spike_objects = []
        self._waveforms = waveforms
        # self._adjusted_labels = []

        # unique = self.get_unique_cluster_labels()
        # if len(unique) < int(max(unique) + 1):
        #     self._format_cluster_count()


    # # e.g. if cluster labels is [1,5,2,4] ==> [0,3,1,2]
    # def _format_cluster_count(self):   
    #     unique = self.get_unique_cluster_labels()
    #     count = len(unique)
    #     adjusted_labels = [i for i in range(count)]
    #     for i in range(len(self._cluster_labels)):
    #         for j in range(len(unique)):
    #             if self._cluster_labels[i] == unique[j]:
    #                 self._cluster_labels[i] = adjusted_labels[j]
    #     self._adjusted_labels = adjusted_labels

    def _read_input_dict(self):
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        # spikes_binary = self._input_dict['spikes_binary']
        cluster_labels = self._input_dict['cluster_labels']
        # assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
        # if len(spikes_binary) > 0:
            # spike_data_present = True
        spike_times = self._input_dict['spike_times']
        assert type(spike_times) == list, 'Spike times are not a list, check inputs'
        # if len(spike_times) > 0:
        #     spike_data_present = True
        # assert spike_data_present == True, 'No spike times or binary spikes provided'
        waveforms = self._extract_waveforms()
        return sample_length, sample_rate, cluster_labels, spike_times, waveforms

    # uses InputKeys() to get channel bychannel waveforms 
    def _extract_waveforms(self):
        input_keys = InputKeys()
        channel_keys = input_keys.get_channel_keys()
        waveforms = []
        for i in range(len(channel_keys)):
            if channel_keys[i] in self._input_dict.keys():
                waveforms.append(self._input_dict[channel_keys[i]])
        return waveforms

    # returns all channel waveforms across all spike times
    def get_all_channel_waveforms(self):
        return self._waveforms
    
    # returns specific channel waveforms across all spike times
    def get_single_channel_waveforms(self, id):
        # assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
        single_channel = []
        for i in range(len(self._spike_times)):
            single_channel.append(self._waveforms[id-1][i])
        return single_channel

    # returns uniqe cluster labels (id of clusters)
    def get_unique_cluster_labels(self):
        visited = []
        for i in range(len(self._cluster_labels)):
            if self._cluster_labels[i] not in visited:
                visited.append(self._cluster_labels[i])
        return sorted(visited)
    
    def get_cluster_labels(self):
        return self._cluster_labels

    def get_single_cluster_firing_rate(self, cluster_id):
        assert cluster_id in self._cluster_labels, 'Invalid cluster ID'
        T = self.timestamps[1] - self.timestamps[0]
        count, _, _ = self.get_single_spike_cluster_instance(cluster_id)
        rate = float(count / T)
        return rate

    # firing rate list, 1 per cluster
    def get_all_cluster_firing_rates(self):
        rates = []
        for i in self.get_unique_cluster_labels():
            rates.append(self.get_single_cluster_firing_rate(i))
        return rates

    # Get specific SpikeCluster() instance
    def get_single_spike_cluster_instance(self, cluster_id):
        # assert cluster_id in self._cluster_labels, 'Invalid cluster ID'
        count = 0
        cluster_spike_times = []
        cluster_waveforms = []
        for i in range(len(self._spike_times)):
            if self._cluster_labels[i] == cluster_id:
                count += 1
                cluster_spike_times.append(self._spike_times[i])
                waveforms_by_channel = []
                for j in range(len(self._waveforms)):
                    waveforms_by_channel.append(self._waveforms[j][i])
                cluster_waveforms.append(waveforms_by_channel)

        assert len(cluster_waveforms[0]) > 0 and len(cluster_waveforms[0]) <= 8
        assert len(cluster_waveforms) == count
        return count, cluster_spike_times, cluster_waveforms

    # List of SpikeCluster() instances
    def get_spike_cluster_instances(self):
        if len(self._spike_clusters) == 0:
            self._make_spike_cluster_instances()
            return self._spike_clusters
        else:
            return self._spike_clusters

    # Spike objects (Spike() + waveforms) for one cluster (list)
    def get_single_spike_cluster_objects(self, cluster_id):
        self._spike_clusters = self.get_spike_cluster_instances()
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            if unique[i] == cluster_id:
                spike_cluster = self._spike_clusters[i]
        spike_object_instances = spike_cluster.get_spike_object_instances()
        return spike_object_instances

    # Spike objects across all clusters (list of list)
    def get_spike_cluster_objects(self):
        instances = []
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            spike_cluster = self._spike_clusters[i]
            instances.append(spike_cluster.get_spike_object_instances())
        return instances

    # Laze eval, called with get_spike_cluster_instances
    def _make_spike_cluster_instances(self):
        # arr to collect SpikeTrain() instances
        instances = []
        # labelled = []
        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in self.get_unique_cluster_labels():
            # if self._cluster_labels[i] not in labelled:
            input_dict = {}
            input_dict['sample_length'] = self._sample_length
            input_dict['sample_rate'] = self._sample_rate
            input_dict['cluster_label'] = i
            if len(self._spike_times) > 0:
                _, cluster_spike_times, _ = self.get_single_spike_cluster_instance(i)
                input_dict['spike_times'] = cluster_spike_times
            else:
                input_dict['spike_times'] = []
            # input_dict['label'] = self._label
            for j in range(len(self._waveforms)):
                key = 'ch' + str(j+1)
                input_dict[key] = self._waveforms[j][i]
            instances.append(SpikeCluster(input_dict))
                # labelled.append(self._cluster_labels[i])
        # assert len(labelled) == max(self._cluster_labels)
        self._spike_clusters = instances
        # print(instances)

class SpikeCluster(): # collection of spike objects
    """
    Class to represent SpikeCluster(). Set of 1D spike times belonging to same cluster
    Will create a collection of spike objects in each cluster

    Similar methods to SpikeClusterBatch() but for a given cluster and not a collection of cluster
    """
    def __init__(self, input_dict):
        self._input_dict = input_dict
        sample_length, sample_rate, cluster_label, spike_times, waveforms = self._read_input_dict()

        # self._make_spike_object_instances()

        assert type(cluster_label) == int, 'Cluster labels missing'
        assert len(spike_times) > 0, 'Spike times missing'
        assert len(waveforms) <= 8 and len(waveforms) > 0, 'Cannot have fewer than 0 or more than 8 channels'

        self.timestamps = make_seconds_index_from_rate(sample_length, sample_rate)
 
        self._spike_times = spike_times
        self._label = cluster_label
        self._sample_length = sample_length
        self._sample_rate = sample_rate
        self._spike_objects = []
        self._waveforms = waveforms

    def get_cluster_firing_rate(self):
        T = self.timestamps[1] - self.timestamps[0]
        rate = float(len(self._spike_times) / T)
        return rate

    def get_cluster_spike_count(self):
        return len(self._spike_times)

    def get_all_channel_waveforms(self):
        return self._waveforms

    # id is channel number. Can be: 1,2,3,4,5,6,7,8
    # all waveforms for a channel (multiple per spike)
    def get_single_channel_waveforms(self, id):
        assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
        return self._waveforms[id-1]

    def get_spike_object_instances(self):
        if len(self._spike_objects) == 0:
            self._make_spike_object_instances()
            return self._spike_objects
        else:
            return self._spike_objects

    def _make_spike_object_instances(self):
        # arr to collect SpikeTrain() instances
        instances = []

        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in range(len(self._spike_times)):
            input_dict = {}
            input_dict['sample_length'] = self._sample_length
            input_dict['sample_rate'] = self._sample_rate
            input_dict['cluster_label'] = self._label
            if len(self._spike_times) > 0:
                input_dict['spike_time'] = self._spike_times[i]
            else:
                input_dict['spike_time'] = []
            input_dict['label'] = self._label
            for j in range(len(self._waveforms)):
                key = 'ch' + str(j+1)
                input_dict[key] = self._waveforms[j][i]
            instances.append(Spike(input_dict))

        self._spike_objects = instances

    def set_cluster_label(self, label):
        if len(self._spike_objects) == 0:
            self._make_spike_object_instances()
        for i in range(len(self._spike_objects)):
            self._spike_objects[i].set_cluster_label(label)
        self._label = label
        print('Cluster label updated for all SpikeObject in cluster')
    
    def get_cluster_label(self):
        return self._label

    def _read_input_dict(self):
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        # spikes_binary = self._input_dict['spikes_binary']
        cluster_label = self._input_dict['cluster_label']
        # assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
        # if len(spikes_binary) > 0:
            # spike_data_present = True
        spike_times = self._input_dict['spike_times']
        assert type(spike_times) == list, 'Spike times are not a list, check inputs'
        if len(spike_times) > 0:
            spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        waveforms = self._extract_waveforms()
        return sample_length, sample_rate, cluster_label, spike_times, waveforms

    def _extract_waveforms(self):
        input_keys = InputKeys()
        channel_keys = input_keys.get_channel_keys()
        waveforms = []
        for i in range(len(channel_keys)):
            if channel_keys[i] in self._input_dict.keys():
                waveforms.append(self._input_dict[channel_keys[i]])
        return waveforms

class SpikeTrainBatch():
    """
    Class to hold collection of 1D spike trains
    """
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
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        spikes_binary = self._input_dict['spikes_binary']
        assert type(spikes_binary) == list, 'Binary spikes are not a list, check inputs'
        if len(spikes_binary) > 0:
            assert type(spikes_binary[0]) == list, 'Binary spikes are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True
        spike_times = self._input_dict['spike_times']
        assert type(spike_times) == list, 'Spike times are not a list, check inputs'
        if len(spike_times) > 0:
            assert type(spike_times[0]) == list, 'Spike times are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        return sample_length, sample_rate, spikes_binary, spike_times

    def _make_spike_train_instance(self):
        # arr to collect SpikeTrain() instances
        instances = []

        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in range(self.units):
            input_dict = {}
            input_dict['sample_length'] = self._sample_length
            input_dict['sample_rate'] = self._sample_rate
            if len(self._spikes_binary) > 0:
                input_dict['spikes_binary'] = self._spikes_binary[i]
            else:
                input_dict['spikes_binary'] = []
            if len(self._spike_times) > 0:
                input_dict['spike_times'] = self._spike_times[i]
            else:
                input_dict['spike_times'] = []
            instances.append(SpikeTrain(input_dict))

        self._spike_train_instances = instances

    def get_spike_train_instances(self):
        if len(self._spike_train_instances) == 0:
            self._make_spike_train_instance()
        return self._spike_train_instances

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


