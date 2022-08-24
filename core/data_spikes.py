

class SpikeTrain():
    """Sorted spike trains
    """
    @abstractmethod
    def __init__(self, spike_ids, spike_times, spike_features):
        pass

    # check if the spike train is in time-stamp format or binary format
    # if in binary format, create binary data object and extract timestamp format as well
    # if in time-stamp format, create time-stamp data object only
    # have method for storing unbinned and binned rate estimates


class SpatialSpikeTrain2D(TimeStampSpikeTrain):
    def __init__():
        pass


class Spike():
    """Spike
    """
    def __init__(self, spike_id, spike_time, spike_feature):
        pass


@dataclass
class SpikeTrain():
    """Spike train
    """
    spike_ids: List[int]
    spike_times: List[str]
    spike_features: List[float]

    def __init__(self, spike_ids, spike_times, spike_features):
        self.spike_ids = spike_ids
        self.spike_times = spike_times
        self.spike_features = spike_features

    def __len__(self):
        return len(self.spike_ids)

    def __getitem__(self, key):
        return self.spike_ids[key], self.spike_times[key], self.spike_features[key]

    def __iter__(self):
        for i in range(len(self.spike_ids)):
            yield self.spike_ids[i], self.spike_times[i], self.spike_features[i]

    def __repr__(self):
        return f'SpikeTrain(spike_ids={self.spike_ids}, spike_times={self.spike_times}, spike_features={self.spike_features})'

    def __str__(self):
        return f'SpikeTrain(spike_ids={self.spike_ids}, spike_times={self.spike_times}, spike_features={self.spike_features})'

    def __eq__(self, other):
        return self.spike_ids == other.spike_ids and self.spike_times == other.spike_times and self.spike_features == other.spike_features

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.spike_ids, self.spike_times, self.spike_features))

    def __lt__(self, other):
        return self.spike_times < other.spike_times

    def __le__(self, other):
        return self.spike_times <= other.spike_times

    def __gt__(self, other):
        return