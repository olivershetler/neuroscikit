import os, sys
import numpy as np
import multiprocessing as mp
import functools
import itertools
import cv2
from numba import jit, njit
import matplotlib.pyplot as plt

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.spatial_spike_train import SpatialSpikeTrain2D
from core.spatial import Position2D


class HaftingOccupancyMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D, **kwargs):
        self.x = spatial_spike_train.x
        self.y = spatial_spike_train.y
        self.t = spatial_spike_train.t
        self.spatial_spike_train = spatial_spike_train
        self.arena_size = spatial_spike_train.arena_size

        self.map_data = None

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata
        
        self.smoothing_factor = self.session_metadata.session_object.smoothing_factor

        if 'smoothing_factor' in kwargs:
            print('overriding session smoothing factor for input smoothing facator')
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
            print('overriding session smoothing factor for input smoothing facator')

    def get_occupancy_map(self, smoothing_factor=None):
        if self.smoothing_factor != None:
            smoothing_factor = self.smoothing_factor
            assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'
        else:
            self.smoothing_factor = smoothing_factor

        self.map_data = self.compute_occupancy_map(self.t, self.x, self.y, self.arena_size, smoothing_factor)

        self.spatial_spike_train.add_map_to_stats('occupancy', self)

        return self.map_data

    def compute_occupancy_map(self, pos_t, pos_x, pos_y, arena_size, smoothing_factor, resolution=64, mask_threshold=1):

        arena_ratio = arena_size[0]/arena_size[1]
        h = smoothing_factor #smoothing factor in centimeters

        if arena_ratio > 1: # if arena height (y) is bigger than width (x)
            x_vec = np.linspace(min(pos_x), max(pos_x), int(resolution))
            y_vec = np.linspace(min(pos_y), max(pos_y), int(resolution*arena_ratio))
        if arena_ratio < 1: # if arena height is is less than width
            x_vec = np.linspace(min(pos_x), max(pos_x), int(resolution/arena_ratio))
            y_vec = np.linspace(min(pos_y), max(pos_y), int(resolution))
        else: # if height == width
            x_vec = np.linspace(min(pos_x), max(pos_x), int(resolution))
            y_vec = np.linspace(min(pos_y), max(pos_y), int(resolution))

        # executor = mp.Pool(mp.cpu_count()) # change this to mp.cpu_count() if you want to use all cores
        futures = list(map(functools.partial(_pos_pdf, pos_x, pos_y, pos_t, smoothing_factor), ((x, y) for x, y in itertools.product(x_vec, y_vec))))
        occupancy_map = np.array(futures).reshape(len(y_vec), len(x_vec))

        mask_values = functools.partial(_mask_points_far_from_curve, mask_threshold, pos_x, pos_y)
        mask_grid = np.array(list(map(mask_values, itertools.product(x_vec, y_vec)))).reshape(len(y_vec), len(x_vec))

        mask_grid = _interpolate_matrix(mask_grid, cv2_interpolation_method=cv2.INTER_NEAREST)
        mask_grid = mask_grid.astype(bool)
        occupancy_map = _interpolate_matrix(occupancy_map, cv2_interpolation_method=cv2.INTER_NEAREST)

        # original, unparallelized code, in case parallelization is causing problems
        #for xi, x in enumerate(x_vec):
            #for yi, y in enumerate(y_vec):
                #occupancy_map[yi,xi] = _pos_pdf((x, y))
                #mask points farther than 4 cm from the curve
                #mask_grid[yi,xi] = _distance_to_curve(point_x=x, point_y=y, curve_x=pos_x, curve_y=pos_y) > 4

        valid_occupancy_map = np.ma.array(occupancy_map, mask=mask_grid)

        valid_occupancy_map = np.rot90(valid_occupancy_map)

        return valid_occupancy_map


class HaftingSpikeMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D, **kwargs):
        # self._input_dict = input_dict
        # self.spatial_spike_train = self._read_input_dict()
        self.spatial_spike_train = spatial_spike_train
        self.spike_x, self.spike_y = self.spatial_spike_train.get_spike_positions()
        self.arena_size = self.spatial_spike_train.arena_size
        self.map_data = None

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata
        
        self.smoothing_factor = self.session_metadata.session_object.smoothing_factor

        if 'smoothing_factor' in kwargs:
            print('overriding session smoothing factor for input smoothing facator')
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
            print('overriding session smoothing factor for input smoothing facator')
    # def _read_input_dict(self):
    #     spatial_spike_train = None
    #     if 'spatial_spike_train' in self._input_dict:
    #         spatial_spike_train = self._input_dict['spatial_spike_train']
    #         assert isinstance(spatial_spike_train, SpatialSpikeTrain2D)
    #     return spatial_spike_train

    def get_spike_map(self, smoothing_factor=None):
        if self.smoothing_factor != None:
            smoothing_factor = self.smoothing_factor
            assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'
        else:
            self.smoothing_factor = smoothing_factor

        self.map_data = self.compute_spike_map(self.spike_x, self.spike_y, smoothing_factor, self.arena_size)

        self.spatial_spike_train.add_map_to_stats('spike', self)

        return self.map_data

    def compute_spike_map(self, spike_x, spike_y, smoothing_factor, arena_size, resolution=64):
        arena_ratio = arena_size[0]/arena_size[1]
        # h = smoothing_factor #smoothing factor in centimeters

        if arena_ratio > 1: # if arena height (y) is bigger than width (x)
            x_vec = np.linspace(min(spike_x), max(spike_x), int(resolution))
            y_vec = np.linspace(min(spike_y), max(spike_y), int(resolution*arena_ratio))
        if arena_ratio < 1: # if arena height is is less than width
            x_vec = np.linspace(min(spike_x), max(spike_x), int(resolution/arena_ratio))
            y_vec = np.linspace(min(spike_y), max(spike_y), int(resolution))
        else: # if height == width
            x_vec = np.linspace(min(spike_x), max(spike_x), int(resolution))
            y_vec = np.linspace(min(spike_y), max(spike_y), int(resolution))


        if resolution >= 170: # This threshold was empirically determined on the development machine. Feel free to change it for other machines.
            # parallelized code for large resolution
            executor = mp.Pool(mp.cpu_count()) # change this to mp.cpu_count() if you want to use all cores
            futures = list(executor.map(functools.partial(_spike_pdf, spike_x, spike_y, smoothing_factor), ((x,y) for x, y in itertools.product(x_vec, y_vec))))
            spike_map = np.array(futures).reshape(len(y_vec), len(x_vec))

        else:
            # non-parallel code is faster for smaller resolutions
            spike_map_vector = [_spike_pdf(spike_x, spike_y, smoothing_factor, (x,y)) for x,y in itertools.product(x_vec,y_vec)]
            spike_map = np.array(spike_map_vector).reshape(len(y_vec), len(x_vec))

        spike_map = np.rot90(spike_map)

        # Resize maps
        spike_map = _interpolate_matrix(spike_map, cv2_interpolation_method=cv2.INTER_NEAREST)

        return spike_map

class HaftingRateMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D, **kwargs):
        
        self.occ_map = spatial_spike_train.get_map('occupancy')
        if self.occ_map == None:
            self.occ_map = HaftingOccupancyMap(spatial_spike_train)
        self.spike_map = spatial_spike_train.get_map('spike')
        if self.spike_map == None:
            self.spike_map = HaftingSpikeMap(spatial_spike_train)
        self.spatial_spike_train = spatial_spike_train
        self.arena_size = spatial_spike_train.arena_size

        assert isinstance(self.occ_map, HaftingOccupancyMap)
        assert isinstance(self.spike_map, HaftingSpikeMap)

        self.map_data = None
        self.raw_map_data = None

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata
        
        self.smoothing_factor = self.session_metadata.session_object.smoothing_factor

        if 'smoothing_factor' in kwargs:
            print('overriding session smoothing factor for input smoothing facator')
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
            print('overriding session smoothing factor for input smoothing facator')

    def get_rate_map(self, smoothing_factor=None):
        if smoothing_factor == None:
            smoothing_factor = self.smoothing_factor
            assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'

        self.map_data, self.raw_map_data = self.compute_rate_map(self.occ_map, self.spike_map)

        self.spatial_spike_train.add_map_to_stats('rate', self)

        return self.map_data, self.raw_map_data

    def compute_rate_map(self, occupancy_map, spike_map):
        '''
        Parameters:
            spike_x: the x-coordinates of the spike events
            spike_y: the y-coordinates of the spike events
            pos_x: the x-coordinates of the position events
            pos_y: the y-coordinates of the position events
            pos_time: the time of the position events
            smoothing_factor: the shared smoothing factor of the occupancy map and the spike map
            arena_size: the size of the arena
        Returns:
            rate_map: spike density divided by occupancy density
        '''
        if self.smoothing_factor != None:
            occ_map_data = occupancy_map.get_occupancy_map(self.smoothing_factor)
            spike_map_data = spike_map.get_spike_map(self.smoothing_factor)
        else:
            print('No smoothing factor provided, proceeding with value of 3')
            occ_map_data = occupancy_map.get_occupancy_map(3)
            spike_map_data = spike_map.get_spike_map(3)
            
        
        
        assert occ_map_data.shape == spike_map_data.shape

        rate_map_raw = _compute_unmasked_ratemap(occ_map_data, spike_map_data)

        rate_map = np.ma.array(rate_map_raw, mask=occ_map_data.mask)
        # rate_map = np.ma.array(rate_map, mask=occ_map_data)

        return rate_map, rate_map_raw

def _compute_unmasked_ratemap(occpancy_map, spike_map):
        return spike_map/occpancy_map

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

def _gaussian2D_pdf(sigma, x, y):
    '''
    Parameters:
        sigma: the standard deviation of the gaussian kernel
        x: the x-coordinate with respect to the center of the distribution
        y: the y-coordinate with respect to the distribution
    Returns:
        z: the 2D PDF value at (x,y)
    '''
    z = 1/(2*np.pi*(sigma**2))*np.exp(-(x**2 + y**2)/(2*(sigma**2)))
    return z

def _get_point_pdfs(rel_pos):
    pdfs = []
    for rel_xi, rel_yi in rel_pos:
        pdfs.append(_gaussian2D_pdf(1, rel_xi, rel_yi))
    return np.array(pdfs)

def _sum_point_pdfs(pdfs):
    return np.sum(pdfs)

def _spike_pdf(spike_x, spike_y, smoothing_factor, point):

    x, y = point

    rel_x = (spike_x - x)/smoothing_factor

    rel_y = (spike_y - y)/smoothing_factor

    rel_coords = ((rel_xi, rel_yi) for rel_xi, rel_yi in zip(rel_x, rel_y))

    pdfs = _get_point_pdfs(rel_coords)

    estimate = _sum_point_pdfs(pdfs)

    return estimate

def _integrate_pos_pdfs(pdfs, pos_time):
    return np.trapz(y=np.array(pdfs).T, x=np.array(pos_time).T)

def _pos_pdf(pos_x: np.ndarray, pos_y: np.ndarray, pos_time: np.ndarray, smoothing_factor, point: tuple, ):

        x, y = point

        rel_x = (pos_x - x)/smoothing_factor

        rel_y = (pos_y - y)/smoothing_factor

        rel_coords = ((rel_xi, rel_yi) for rel_xi, rel_yi in zip(rel_x, rel_y))

        pdfs = _get_point_pdfs(rel_coords)

        estimate = _integrate_pos_pdfs(pdfs, pos_time)

        return float(estimate)

def _mask_points_far_from_curve(mask_threshold, curve_x, curve_y, point):
    '''
    Parameters:
        point (numeric): the point to calculate the distance to the curve
        curve (iterable): the curve to calculate the distance to the point
    Returns:
        distance (numeric): the distance between the point and the curve

    This is very inefficient --- O(base_resolution^2 * curve_length) --- any way to speed this up?
    '''
    point_x, point_y = point
    distance = min(np.sqrt((point_x - curve_x)**2 + (point_y - curve_y)**2))
    if distance > mask_threshold:
        return 1
    else:
        return 0

    return distance

def save_map(occupancy_map, title, units_label, file_name, directory):
    '''
    Parameters:
        map: the map to save
        filename: the filename to save the map as
    '''
    # print(np.max(occupancy_map))
    f, ax = plt.subplots(figsize=(7, 7))
    m = ax.imshow(occupancy_map, cmap="jet", interpolation="nearest")
    cbar = f.colorbar(m, ax=ax, shrink=0.8)
    cbar.set_label(units_label, rotation=270, labelpad=20)
    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(directory + '/' + file_name + '.png')
    plt.close('all')

