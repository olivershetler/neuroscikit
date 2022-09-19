import numpy as np

from core.spatial import Location
from core.spikes import SpikeTrain


class SpatialSpikeTrain2D():

    def __init__(self, spike_train: SpikeTrain, location: Location):
        self.spikt_times = spike_train.times

        self.x, self.y = get_spike_positions(self.spikt_times, location.x, location.y, location.t)

        assert len(self.x) == len(self.y) == len(self.spikt_times)


    #def spike_pos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):
    def get_spike_positions(self, spike_times, x, y, t):

        delta_t = t[1] - t[0]
        #assert np.all(np.diff(t) == delta_t)
        def _match(time, time_index, spike_time):
            if spiek_time >= time and spike_time < time + delta_t:
                return index

        spike_index = list(filter(_match, t, range(len(t)), spike_times))

        return x[spike_index], y[spike_index]


        def shuffle_spike_positions(self, displacement):
            pass




