from core.spikes import SpikeTrain
from core.spatial import Location
from library.maps.spatial_spike_train import SpatialSpikeTrain2D


def test_spatial_spike_train():
    spike_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    spike_train = SpikeTrain(spike_times)
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    location = Location(x, y, t)

    spatial_spike_train = SpatialSpikeTrain2D(spike_train, location)

    assert len(spatial_spike_train.x) == len(spatial_spike_train.y) == len(spatial_spike_train.spikt_times)
    assert spatial_spike_train.x == x
    assert spatial_spike_train.y == y
    assert spatial_spike_train.spikt_times == spike_times