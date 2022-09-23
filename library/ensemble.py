import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from core.spikes import * 


class SpikeTrainBatch():
    """
    Class to hold collection of 1D spike trains
    """
    def __init__(self, input_dict):
        self._input_dict = input_dict

        duration, sample_rate, events_binary, event_times = self._read_input_dict()

        # self.time_index = make_seconds_index_from_rate(duration, sample_rate)

        assert ((len(events_binary) == 0) and (len(event_times) == 0)) != True, "No spike data provided"

        self.units = max(len(events_binary), len(event_times))

        self.sample_rate = sample_rate
        self.duration = duration
        self.event_times = event_times
        self.events_binary = events_binary
        self.event_labels = []
        self._event_rate = None
        self._spike_train_instances = []

    def _read_input_dict(self):
        duration = self._input_dict['duration']
        sample_rate = self._input_dict['sample_rate']
        events_binary = self._input_dict['events_binary']
        assert type(events_binary) == list, 'Binary spikes are not a list, check inputs'
        if len(events_binary) > 0:
            assert type(events_binary[0]) == list, 'Binary spikes are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True
        event_times = self._input_dict['event_times']
        assert type(event_times) == list, 'Spike times are not a list, check inputs'
        if len(event_times) > 0:
            assert type(event_times[0]) == list, 'Spike times are not nested lists (2D), check inputs are not 1D'
            spike_data_present = True
        assert spike_data_present == True, 'No spike times or binary spikes provided'
        return duration, sample_rate, events_binary, event_times

    def _make_spike_train_instance(self):
        # arr to collect SpikeTrain() instances
        instances = []

        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in range(self.units):
            input_dict = {}
            input_dict['duration'] = self.duration
            input_dict['sample_rate'] = self.sample_rate
            if len(self.events_binary) > 0:
                input_dict['events_binary'] = self.events_binary[i]
            else:
                input_dict['events_binary'] = []
            if len(self.event_times) > 0:
                input_dict['event_times'] = self.event_times[i]
            else:
                input_dict['event_times'] = []
            instances.append(SpikeTrain(input_dict))

        self._spike_train_instances = instances

    def get_spike_train_instances(self):
        if len(self._spike_train_instances) == 0:
            self._make_spike_train_instance()
        return self._spike_train_instances

    def get_indiv_event_rate(self):
        self.get_spike_train_instances()
        event_rates = []
        for i in range(len(self._spike_train_instances)):
            event_rates.append(self._spike_train_instances[i].get_event_rate())
        return event_rates

    def get_average_event_rate(self):
        self.get_spike_train_instances()
        event_rates = self.get_indiv_event_rate()
        return sum(event_rates) / len(event_rates)

    def _set_binary(self):
        for i in range(len(self._spike_train_instances)):
            self.events_binary.append(self._spike_train_instances[i].get_binary())
        return self.events_binary

    def _set_event_times(self):
        for spike_train in self._spike_train_instances:
            self.event_times.append(spike_train.get_event_times())
        return self.event_times

    # get 2d spike inputs as binary
    def get_binary(self):
        if len(self.events_binary) == 0:
            self.get_spike_train_instances()
            self._set_binary()
        else:
            return self.events_binary

    # get 2d spike inputs as time_index
    def get_event_times(self):
        if len(self.event_times) == 0:
            self.get_spike_train_instances()
            self._set_event_times()
        else:
            return self.event_times




class SpikeClusterBatch():
    """
    Class to batch process SpikeClusters. Can pass in unorganized set of 1D spike times + cluster labels
    will create a collection of spike clusters with a collection of spikie objects in each cluster
    """
    def __init__(self, input_dict):
        self._input_dict = input_dict
        duration, sample_rate, cluster_labels, event_times, waveforms = self._read_input_dict()

        assert type(cluster_labels) == list, 'Cluster labels missing'
        assert type(cluster_labels[0]) == int, 'Cluster labels missing'
        assert len(event_times) > 0, 'Spike times missing'
        assert len(waveforms) <= 8 and len(waveforms) > 0, 'Cannot have fewer than 0 or more than 8 channels'

        self.time_index = make_seconds_index_from_rate(duration, sample_rate)

        self.event_times = event_times
        self.cluster_labels = cluster_labels
        self.duration = duration
        self.sample_rate = sample_rate
        self.spike_clusters = []
        self.spike_objects = []
        self.waveforms = waveforms
        # self._adjusted_labels = []

        # unique = self.get_unique_cluster_labels()
        # if len(unique) < int(max(unique) + 1):
        #     self._format_cluster_count()


    # # e.g. if cluster labels is [1,5,2,4] ==> [0,3,1,2]
    # def _format_cluster_count(self):
    #     unique = self.get_unique_cluster_labels()
    #     count = len(unique)
    #     adjusted_labels = [i for i in range(count)]
    #     for i in range(len(self.cluster_labels)):
    #         for j in range(len(unique)):
    #             if self.cluster_labels[i] == unique[j]:
    #                 self.cluster_labels[i] = adjusted_labels[j]
    #     self._adjusted_labels = adjusted_labels

    def _read_input_dict(self):
        duration = self._input_dict['duration']
        sample_rate = self._input_dict['sample_rate']
        # events_binary = self._input_dict['events_binary']
        cluster_labels = self._input_dict['event_labels']
        # assert type(events_binary) == list, 'Binary spikes are not a list, check inputs'
        # if len(events_binary) > 0:
            # spike_data_present = True
        event_times = self._input_dict['event_times']
        assert type(event_times) == list, 'Spike times are not a list, check inputs'
        # if len(event_times) > 0:
        #     spike_data_present = True
        # assert spike_data_present == True, 'No spike times or binary spikes provided'
        waveforms = self._extract_waveforms()
        return duration, sample_rate, cluster_labels, event_times, waveforms

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
        return self.waveforms

    # returns specific channel waveforms across all spike times
    def get_single_channel_waveforms(self, id):
        # assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
        single_channel = []
        for i in range(len(self.event_times)):
            single_channel.append(self.waveforms[id-1][i])
        return single_channel

    # returns uniqe cluster labels (id of clusters)
    def get_unique_cluster_labels(self):
        visited = []
        for i in range(len(self.cluster_labels)):
            if self.cluster_labels[i] not in visited:
                visited.append(self.cluster_labels[i])
        return sorted(visited)

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_single_cluster_firing_rate(self, cluster_id):
        assert cluster_id in self.cluster_labels, 'Invalid cluster ID'
        T = self.time_index[1] - self.time_index[0]
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
        # assert cluster_id in self.cluster_labels, 'Invalid cluster ID'
        count = 0
        clusterevent_times = []
        cluster_waveforms = []
        for i in range(len(self.event_times)):
            if self.cluster_labels[i] == cluster_id:
                count += 1
                clusterevent_times.append(self.event_times[i])
                waveforms_by_channel = []
                for j in range(len(self.waveforms)):
                    waveforms_by_channel.append(self.waveforms[j][i])
                cluster_waveforms.append(waveforms_by_channel)

        assert len(cluster_waveforms[0]) > 0 and len(cluster_waveforms[0]) <= 8
        assert len(cluster_waveforms) == count
        return count, clusterevent_times, cluster_waveforms

    # List of SpikeCluster() instances
    def get_spike_cluster_instances(self):
        if len(self.spike_clusters) == 0:
            self._make_spike_cluster_instances()
            return self.spike_clusters
        else:
            return self.spike_clusters

    # Spike objects (Spike() + waveforms) for one cluster (list)
    def get_single_spike_cluster_objects(self, cluster_id):
        self.spike_clusters = self.get_spike_cluster_instances()
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            if unique[i] == cluster_id:
                spike_cluster = self.spike_clusters[i]
        spike_object_instances = spike_cluster.get_spike_object_instances()
        return spike_object_instances

    # Spike objects across all clusters (list of list)
    def get_spike_cluster_objects(self):
        instances = []
        unique = self.get_unique_cluster_labels()
        for i in range(len(unique)):
            spike_cluster = self.spike_clusters[i]
            instances.append(spike_cluster.get_spike_object_instances())
        return instances

    # Laze eval, called with get_spike_cluster_instances
    def _make_spike_cluster_instances(self):
        # arr to collect SpikeTrain() instances
        instances = []
        # labelled = []
        # Both are 2d arrays so can do len() to iterate thru number of cells
        for i in self.get_unique_cluster_labels():
            # if self.cluster_labels[i] not in labelled:
            input_dict = {}
            input_dict['duration'] = self.duration
            input_dict['sample_rate'] = self.sample_rate
            input_dict['cluster_label'] = i
            if len(self.event_times) > 0:
                _, clusterevent_times, _ = self.get_single_spike_cluster_instance(i)
                input_dict['event_times'] = clusterevent_times
            else:
                input_dict['event_times'] = []
            # input_dict['label'] = self.label
            for j in range(len(self.waveforms)):
                key = 'channel_' + str(j+1)
                input_dict[key] = self.waveforms[j][i]
            instances.append(SpikeCluster(input_dict))
                # labelled.append(self.cluster_labels[i])
        # assert len(labelled) == max(self.cluster_labels)
        self.spike_clusters = instances



