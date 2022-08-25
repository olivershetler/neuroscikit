import numpy as np

class SpikeTrain():
    """Single spike train
    """
    spikes_raw: list[int]
    spike_times: list[float]
    spike_ids: list[int]
    spike_features: list[float]

    def __init__(self,  timestamps, spike_features=[], spikes_raw=[], spike_times=[]):
        self.timestamps = timestamps

        assert ((len(spikes_raw) > 0) and (len(spike_times) > 0)) != True, "No spike data provided"

        spike_ids = []

        # binary <-- time stamps
        if len(spikes_raw) == 0:
            spikes_raw = []

            for i in range(len(self.timestamps)):
                if self.timestamps[i] in spike_times:
                    spike_ids.append(i)
                    spikes_raw.append(1)
                else:
                    spikes_raw.append(0)

        # binary --> time stamps
        if len(spike_times) == 0:
            spike_times = []

            for i in range(len(self.timestamps)):
                if spikes_raw[i] == 1:
                    spike_ids.append(i)
                    spike_times.append(self.timestamps[i])

        assert sum(spikes_raw) == len(spike_times)

        self.spike_times = spike_times
        self.spikes_raw = spikes_raw
        self.spike_ids = spike_ids
        self.spike_features = spike_features
        self._spike_rate = None

    def __len__(self):
        return len(self.spike_ids)

    def __getitem__(self, key):
        return self.spike_ids[key], self.spike_times[key], self.spikes_raw[key]

    def __iter__(self):
        for i in range(len(self.spike_ids)):
            yield self.spike_ids[i], self.spike_times[i], self.spikes_raw[i]

    def __repr__(self):
        return f'SpikeTrain(spike_ids={self.spike_ids}, spike_times={self.spike_times}, spikes_raw={self.spikes_raw})'

    def __str__(self):
        return f'SpikeTrain(spike_ids={self.spike_ids}, spike_times={self.spike_times}, spikes_raw={self.spikes_raw})'

    def __eq__(self, other):
        return self.spike_ids == other.spike_ids and self.spike_times == other.spike_times and self.spikes_raw == other.spikes_raw

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.spike_ids, self.spike_times, self.spikes_raw))

    def __lt__(self, other):
        return self.spike_times < other.spike_times

    def __le__(self, other):
        return self.spike_times <= other.spike_times

    def __gt__(self, other):
        return self.spike_times > other.spike_times

    def __ge__(self, other):
        return self.spike_times >= other.spike_times

    def spike_rate(self):
        if self._spike_rate == None:
            T = (self.timestamps[-1] - self.timestamps[0])
            self._spike_rate = float(sum(self.spikes_raw) / T)
            # assert self._spike_rate == float(len(self.spike_times) / T), 'binary vs timestamps returning different spike rate'
            return self._spike_rate
        else:
            return self._spike_rate

    


    


    # check if the spike train is in time-stamp format or binary format
    # if in binary format, create binary data object and extract timestamp format as well
    # if in time-stamp format, create time-stamp data object onlys
    # have method for storing unbinned and binned rate estimatess

# class TimeStampSpikeTrain(SpikeTrain):
#     """ Spike train as time stamps 
#     """

#     def __init(self, spike_times, spike_features):
#         self.spike_times = spike_times

# class BinarySpikeTrain(SpikeTrain):
#     """ Spike train as binary 
#     """

#     def __init(self, spikes_raw, spike_features):
#         self.spikes_raw = spikes_raw



# class SpatialSpikeTrain2D(TimeStampSpikeTrain):
#     def __init__():
#         pass


# class Spike():
#     """Spike
#     """
#     def __init__(self, spike_id, spike_time, spike_feature):
#         pass

