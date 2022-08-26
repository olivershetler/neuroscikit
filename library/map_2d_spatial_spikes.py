


class SpikeTrainPoints(TimeStampSpikeTrain):
    """Sorted spike trains
    """
    def __init__(self, spike_ids, spike_times, spike_x, spike_y):
        pass


class RateMap():
    """
    """
    def __init__(self, spike_train, bin_size):
        pass


class  SpatialRateMap(RateMap):
    """
    Abstract spatial rate map class
    """
    pass

class SpatialRateMap1D(SpatialRateMap):
    """
    Spatial 1D rate map
    """
    pass

class  SpatialRateMap2D(SpatialRateMap):
    """
    Spatial 2D rate map
    """
    def __init__(self, **kwargs):
        pass

    def compute_ratemap(self, TimeStampSpikeTrain):
        """
        Compute rate map
        """
        pass

    def compute_error_map(self, TimeStampSpikeTrain):
        """
        This function will use the rate map as a basis for predicting
        the firing rate of the neurons. Then, the temporally computed
        firing rate for points within the spatial bins will be compared
        and a head map of the differences will be generated.
        """
        pass


class  SpatialRateMap3D(SpatialRateMap):
    """
    Spatial 3D rate map
    """
    def __init__(self, rate_map, **kwargs):
        pass

