

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


class Spike():
    """Spike
    """
    def __init__(self, spike_id, spike_time, spike_feature):
        pass