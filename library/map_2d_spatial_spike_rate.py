from ..library.spike_maps import


class SpikeTrainPoints(TimeStampSpikeTrain):
    """Sorted spike trains
    """
    def __init__(self, spike_ids, spike_times, spike_x, spike_y):


class RateMap():
    """
    """


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


class  SpatialRateMap3D(SpatialRateMap):
    """
    Spatial 3D rate map
    """
    def __init__(self, rate_map, **kwargs):
        pass

