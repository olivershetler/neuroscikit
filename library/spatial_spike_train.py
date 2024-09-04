from turtle import pos
import numpy as np
import os, sys


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spatial import Position2D
from core.subjects import SessionMetadata
from library.ensemble_space import Cell
from core.spikes import SpikeTrain
from library.hafting_spatial_maps import HaftingOccupancyMap, HaftingRateMap, HaftingSpikeMap









# class SpatialSpikeTrain2D():

#     def __init__(self, input_dict: dict, **kwargs):
#         # spike_train: SpikeTrain, position: Position2D, **kwargs):
#         self._input_dict = input_dict
#         self.spike_obj, self.position = self._read_input_dict()
#         self.spike_times = self.spike_obj.event_times
#         self.t, self.x, self.y = self.position.t, self.position.x, self.position.y

#         assert len(self.t) == len(self.x) == len(self.y)

#         if 'session_metadata' in kwargs:
#             self.session_metadata = kwargs['session_metadata']
#         else:
#             self.session_metadata = None
#         if self.position.arena_height != None and self.position.arena_width != None:
#             self.arena_size = (self.position.arena_height, self.position.arena_width)
#         else:
#             self.arena_size = None

#         if 'speed_bounds' in kwargs:
#             self.speed_bounds = kwargs['speed_bounds']
#         else:
#             self.speed_bounds = (0, 100)


#         self.spike_x, self.spike_y, self.new_spike_times = self.get_spike_positions()

#         print('noo')
#         print(len(self.x), len(self.y), len(self.t))
#         print(len(self.spike_x), len(self.spike_y), len(self.new_spike_times))
#         assert len(self.spike_x) == len(self.spike_y) == len(self.new_spike_times)

#         self.stats_dict = self._init_stats_dict()

#     def _read_input_dict(self):
#         spike_obj = None
#         position = None

#         assert ('spike_train' not in self._input_dict and 'cell' in self._input_dict) or ('spike_train' in self._input_dict and 'cell' not in self._input_dict)

#         if 'spike_train' in self._input_dict:
#             spike_obj = self._input_dict['spike_train']
#             assert isinstance(spike_obj, SpikeTrain)
#         elif 'cell' in self._input_dict:
#             spike_obj = self._input_dict['cell']
#             assert isinstance(spike_obj, Cell)

#         if 'position' in self._input_dict:
#             position = self._input_dict['position']
#             assert isinstance(position, Position2D)

#         return spike_obj, position


#     def _init_stats_dict(self):
#         stats_dict = {}

#         map_names = ['autocorr', 'binary', 'spatial_tuning', 'pos_vs_speed', 'rate_vs_time', 'hafting', 'occupancy', 'rate', 'spike', 'map_blobs']

#         for key in map_names:
#             stats_dict[key] = None

#         return stats_dict

#     def add_map_to_stats(self, map_name, map_class):
#         # print(self.stats_dict)
#         assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
#         # assert type(map_class) != np.ndarray and type(map_class) != list
#         self.stats_dict[map_name] = map_class

#     def get_map(self, map_name):
#         assert map_name in self.stats_dict, 'check valid map types to add to stats dict, map type not in stats dict'
#         return self.stats_dict[map_name]

#     #def spike_pos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):
#     def get_spike_positions(self):
#         # if type(self.spike_times) == list:
#         #     spike_array = np.array(self.spike_times)
#         # else:
#         #     spike_array = self.spike_times
#         # time_step_t = self.t[1] - self.t[0]
#         # spike_index = []
#         # for i in range(len(self.t)):
#         #     id_set_1 = np.where(spike_array >= self.t[i])[0]
#         #     id_set_2 = np.where(spike_array < self.t[i] + time_step_t)[0]
#         #     for id in id_set_1:
#         #         if id in id_set_2 and id not in spike_index:
#         #             spike_index.append(id)

#         # # def _match(time, time_index, spike_time):
#         # #     if spike_time >= time and spike_time < time + time_step_t:
#         # #         return time_index

#         # # spike_index = list(filter(_match(self.t, range(len(self.t)), self.spike_times)))
#         # return np.array(self.x)[spike_index], np.array(self.y)[spike_index]

#         v = _speed2D(self.x, self.y, self.t)
#         x, y, t = _speed_bins(self.speed_bounds[0], self.speed_bounds[1], v, self.x, self.y, self.t)

#         cPost = np.copy(t)

#         N = len(self.spike_times)
#         spike_positions_x = np.zeros((N, 1))
#         spike_positions_y = np.zeros_like(spike_positions_x)
#         new_spike_times = np.zeros_like(spike_positions_x)
#         count = -1 # need to subtract 1 because the python indices start at 0 and MATLABs at 1


#         for index in range(N):

#             tdiff = (t -self.spike_times[index])**2
#             tdiff2 = (cPost-self.spike_times[index])**2
#             m = np.amin(tdiff)
#             ind = np.where(tdiff == m)[0]

#             m2 = np.amin(tdiff2)
#             #ind2 = np.where(tdiff2 == m2)[0]

#             if m == m2:
#                 count += 1
#                 spike_positions_x[count] = x[ind[0]]
#                 spike_positions_y[count] = y[ind[0]]
#                 new_spike_times[count] = self.spike_times[index]

#         spike_positions_x = spike_positions_x[:count + 1]
#         spike_positions_y = spike_positions_y[:count + 1]
#         new_spike_times = new_spike_times[:count + 1]

#         return spike_positions_x.flatten(), spike_positions_y.flatten(), new_spike_times.flatten()

# def _speed_bins(lower_speed: float, higher_speed: float, pos_v: np.ndarray,
#                pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple:

#     '''
#         Selectively filters position values of subject travelling between
#         specific speed limits.

#         Params:
#             lower_speed (float):
#                 Lower speed bound (cm/s)
#             higher_speed (float):
#                 Higher speed bound (cm/s)
#             pos_v (np.ndarray):
#                 Array holding speed values of subject
#             pos_x, pos_y, pos_t (np.ndarray):
#                 X, Y coordinate tracking of subject and timestamps

#         Returns:
#             Tuple: new_pos_x, new_pos_y, new_pos_t
#             --------
#             new_pos_x, new_pos_y, new_pos_t (np.ndarray):
#                 speed filtered x,y coordinates and timestamps
#     '''

#     # Initialize empty array that will only be populated with speed values within
#     # specified bounds
#     choose_array = []

#     # Iterate and select speeds
#     for index, element in enumerate(pos_v):
#         if element > lower_speed and element < higher_speed:
#             choose_array.append(index)

#     # construct new x,y and t arrays
#     new_pos_x = np.asarray([ float(pos_x[i]) for i in choose_array])
#     new_pos_y = np.asarray([ float(pos_y[i]) for i in choose_array])
#     new_pos_t = np.asarray([ float(pos_t[i]) for i in choose_array])

#     return new_pos_x, new_pos_y, new_pos_t


# def _speed2D(x, y, t):
#     """calculates an averaged/smoothed speed"""

#     N = len(x)
#     v = np.zeros((N, 1))

#     for index in range(1, N-1):
#         v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
#         t[index + 1] - t[index - 1])

#     v[0] = v[1]
#     v[-1] = v[-2]

#     return v

#         # def shuffle_spike_positions(self, displacement):
#         #     pass

#     # def make_rate_map(self):
#     #     HaftingRateMap(self)

#     # def make_occupancy_map(self):
#     #     HaftingOccupancyMap(self)

#     # def make_spike_map(self):
#     #     HaftingSpikeMap(self)



