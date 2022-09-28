from turtle import pos
import numpy as np
import os, sys

from core.subjects import SessionMetadata

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D
from core.spikes import SpikeTrain


class SpatialSpikeTrain2D():

    def __init__(self, input_dict: dict, **kwargs):
        # spike_train: SpikeTrain, position: Position2D, **kwargs):
        self._input_dict = input_dict
        self.spike_train_instance, self.position = self._read_input_dict()
        self.spike_times = self.spike_train_instance.event_times
        self.t, self.x, self.y = self.position.t, self.position.x, self.position.y
        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        if self.position.arena_height != None and self.position.arena_width != None:
            self.arena_size = (self.position.arena_height, self.position.arena_width)

        self.spike_x, self.spike_y = self.get_spike_positions()

        assert len(self.spike_x) == len(self.spike_y) == len(self.spike_times)

        self.stats_dict = self._init_stats_dict()

    def _read_input_dict(self):
        spike_train = None 
        position = None 
        
        if 'spike_train' in self._input_dict:
            spike_train = self._input_dict['spike_train']
            assert isinstance(spike_train, SpikeTrain)
        if 'position' in self._input_dict:
            position = self._input_dict['position']
            assert isinstance(position, Position2D)

        return spike_train, position


    def _init_stats_dict(self):
        stats_dict = {}

        map_names = ['autocorr', 'binary', 'spatial_tuning', 'pos_vs_speed', 'rate_vs_time', 'hafting', 'occupancy', 'rate', 'spike', 'map_blobs']

        for key in map_names:
            stats_dict[key] = None

        return stats_dict

    def add_map_to_stats(self, map_name, map_class):
        # print(self.stats_dict)
        assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
        # assert type(map_class) != np.ndarray and type(map_class) != list
        self.stats_dict[map_name] = map_class

    def get_map(self, map_name):
        assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
        return self.stats_dict[map_name]

    #def spike_pos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):
    def get_spike_positions(self):
        spike_array = np.array(self.spike_times)
        delta_t = self.t[1] - self.t[0]
        spike_index = []
        for i in range(len(self.t)):
            id_set_1 = np.where(spike_array >= self.t[i])[0]
            id_set_2 = np.where(spike_array < self.t[i] + delta_t)[0]
            for id in id_set_1:
                if id in id_set_2 and id not in spike_index:
                    spike_index.append(id)

        # def _match(time, time_index, spike_time):
        #     if spike_time >= time and spike_time < time + delta_t:
        #         return time_index

        # spike_index = list(filter(_match(self.t, range(len(self.t)), self.spike_times)))
        return np.array(self.x)[spike_index], np.array(self.y)[spike_index]


        # def shuffle_spike_positions(self, displacement):
        #     pass

    # def make_rate_map(self):
    #     HaftingRateMap(self)

    # def make_occupancy_map(self):
    #     HaftingOccupancyMap(self)

    # def make_spike_map(self):
    #     HaftingSpikeMap(self)



