from turtle import pos
import numpy as np
import os, sys

from core.subjects import SessionMetadata 

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D
from core.spikes import SpikeTrain


class SpatialSpikeTrain2D():

    def __init__(self, spike_train: SpikeTrain, position: Position2D, **kwargs):
        self.spike_train_instance = spike_train
        self.spike_times = spike_train.event_times
        self.t, self.x, self.y = position.t, position.x, position.y
        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        if position.arena_height != None and position.arena_width != None:
            self.arena_size = (position.arena_height, position.arena_width)

        self.spike_x, self.spike_y = self.get_spike_positions()

        assert len(self.x) == len(self.y) == len(self.spike_times)

        self.stats_dict = {}
        self._init_stats_dict()

    def _init_stats_dict(self):
        stats_dict = {}

        map_names = ['autocorr', 'binary', 'direction', 'pos_vs_speed', 'rate_vs_time', 'hafting', 'occupancy', 'rate', 'spike']

        for map_name in map_names:
            stats_dict[map_name] = {}

        return stats_dict

    def add_map_to_stats(self, map_name, map_data):
        assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
        self.stats_dict[map_name] = map_data

    def get_map(self, map_name):
        assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
        return self.stats_dict[map_name]

    #def spike_pos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):
    def get_spike_positions(self):

        delta_t = self.t[1] - self.t[0]
        #assert np.all(np.diff(t) == delta_t)
        spike_index = []
        for i in range(len(self.spike_times)):
            if self.spike_times[i] >= self.t[i] and self.spike_times[i] < self.t[i] + delta_t:
                spike_index.append(i)


        # def _match(time, time_index, spike_time):
        #     if spike_time >= time and spike_time < time + delta_t:
        #         return time_index

        # spike_index = list(filter(_match(self.t, range(len(self.t)), self.spike_times)))

        return np.asarray(self.x)[spike_index], np.asarray(self.y)[spike_index]


        # def shuffle_spike_positions(self, displacement):
        #     pass




