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

from library.maps.spatial_spike_train import SpatialSpikeTrain2D
from core.spatial import Position2D


class HaftingOccupancyMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D, **kwargs):
        self.x = spatial_spike_train.x
        self.y = spatial_spike_train.y
        self.t = spatial_spike_train.t
        self.arena_size = spatial_spike_train.arena_size

        self.occupancy_map = None

        self.session_metadata = None
        self.smoothing_factor = None
        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']

    def get_occupancy_map(self, smoothing_factor=None):
        if smoothing_factor == None:
            smoothing_factor = self.smoothing_factor
            assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'

        self.occupancy_map = self.compute_occupancy_map(self.t, self.x, self.y, self.arena_size, self.smoothing_factor)

        return self.occupancy_map

    @staticmethod
    def compute_occupancy_map(pos_t, pos_x, pos_y, arena_size, smoothing_factor, resolution=64, mask_threshold=1):

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

        executor = mp.Pool(mp.cpu_count()) # change this to mp.cpu_count() if you want to use all cores
        futures = list(executor.map(functools.partial(_pos_pdf, pos_x, pos_y, pos_t, smoothing_factor), ((x, y) for x, y in itertools.product(x_vec, y_vec))))
        occupancy_map = np.array(futures).reshape(len(y_vec), len(x_vec))

        mask_values = functools.partial(_mask_points_far_from_curve, mask_threshold, pos_x, pos_y)
        mask_grid = np.array(list(executor.map(mask_values, itertools.product(x_vec, y_vec)))).reshape(len(y_vec), len(x_vec))

        mask_grid = _interpolate_matrix(mask_grid, cv2_interpolation_method=cv2.INTER_NEAREST)
        mask_grid = mask_grid.astype(np.bool)
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
        self.spike_x, self.spike_y = spatial_spike_train.get_spike_positions()

        self.spike_map = None

        self.session_metadata = None
        self.smoothing_factor = None
        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']

    def get_spike_map(self):
        if smoothing_factor == None:
            smoothing_factor = self.smoothing_factor
            assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'

        self.spike_map = self.compute_spike_map(self.spike_x, self.spike_y, self.smoothing_factor)

        return self.spike_map

    def compute_spike_map(spike_x, spike_y, smoothing_factor, arena_size, resolution=64):
        arena_ratio = arena_size[0]/arena_size[1]
        h = smoothing_factor #smoothing factor in centimeters

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
            spike_map_vector = [_spike_pdf(spike_x, spike_y, h, (x,y)) for x,y in itertools.product(x_vec,y_vec)]
            spike_map = np.array(spike_map_vector).reshape(len(y_vec), len(x_vec))

        spike_map = np.rot90(spike_map)

        # Resize maps
        spike_map = _interpolate_matrix(spike_map, cv2_interpolation_method=cv2.INTER_NEAREST)

        return spike_map

class HaftingRateMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D, **kwargs):
        
        self.occupancy_map = spatial_spike_train.get_map('occupancy')
        self.spike_map = spatial_spike_train.get_map('spike')

        assert isinstance(self.occupancy_map, HaftingOccupancyMap)
        assert isinstance(self.spike_map, HaftingSpikeMap)

        self.ratemap = None

        self.session_metadata = None
        self.smoothing_factor = None
        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']

    def get_rate_map(self):
        self.ratemap = self.compute_rate_map(self.occupancy_map, self.spike_map)
        return self.ratemap

    def compute_rate_map(occupancy_map, spike_map):
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
        assert occupancy_map.shape == spike_map.shape

        rate_map = _compute_unmasked_ratemap(occupancy_map, spike_map)

        rate_map = np.ma.array(rate_map, mask=occupancy_map.mask)

        return rate_map

@njit
def _compute_unmasked_ratemap(occpancy_map, spike_map):
        return spike_map/occpancy_map

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

@njit
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

@njit
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

@njit
def _integrate_pos_pdfs(pdfs, pos_time):
    return np.trapz(y=pdfs.T, x=pos_time.T)


def _pos_pdf(pos_x: np.ndarray, pos_y: np.ndarray, pos_time: np.ndarray, smoothing_factor, point: tuple, ):

        x, y = point

        rel_x = (pos_x - x)/smoothing_factor

        rel_y = (pos_y - y)/smoothing_factor

        rel_coords = ((rel_xi, rel_yi) for rel_xi, rel_yi in zip(rel_x, rel_y))

        pdfs = _get_point_pdfs(rel_coords)

        estimate = _integrate_pos_pdfs(pdfs, pos_time)

        return float(estimate)

@njit
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
    print(np.max(occupancy_map))
    f, ax = plt.subplots(figsize=(7, 7))
    m = ax.imshow(occupancy_map, cmap="jet", interpolation="nearest")
    cbar = f.colorbar(m, ax=ax, shrink=0.8)
    cbar.set_label(units_label, rotation=270, labelpad=20)
    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(directory + '/' + file_name + '.png')
    plt.close('all')

