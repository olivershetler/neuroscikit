import os
import sys


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


import numpy as np
import cv2
from PIL import Image
from library.maps.map_utils import _compute_resize_ratio, _interpolate_matrix
from opexebo.analysis import autocorrelation as opexebo_autocorrelation
from library.spatial_spike_train import SpatialSpikeTrain2D
from library.hafting_spatial_maps import HaftingRateMap


# def autocorrelation(ratemap: np.ndarray, arena_size: tuple) -> np.ndarray:
def autocorrelation(spatial_map: SpatialSpikeTrain2D | HaftingRateMap, **kwargs):
    '''
        Compute the autocorrelation map from ratemap

        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.
            pos_x, pos_y (np.ndarray):
                Arrays  tracking x and y coordinates of subject movement
            arena_size (tuple):
                Width and length of tracking arena

        Returns:
            np.ndarray:
                autocorr_OPEXEBO
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = 3

    if isinstance(spatial_map, HaftingRateMap):
        ratemap = spatial_map.get_rate_map(smoothing_factor)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        rate_obj = spatial_map.get_map('rate')
        if rate_obj == None:
            ratemap = HaftingRateMap(spatial_map).get_rate_map(smoothing_factor)
        else:
            ratemap = rate_obj.get_rate_map(smoothing_factor)

    arena_size = spatial_map.arena_size

    x_resize, y_resize = _compute_resize_ratio(arena_size)
    autocorr_OPEXEBO = opexebo_autocorrelation(ratemap)
    autocorr_OPEXEBO = _interpolate_matrix(autocorr_OPEXEBO, cv2_interpolation_method=cv2.INTER_NEAREST) #_resize_numpy2D(autocorr_OPEXEBO, x_resize, y_resize)

    if isinstance(spatial_map, HaftingRateMap):
        spatial_map.spatial_spike_train.add_map_to_stats('autocorr', autocorr_OPEXEBO)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        spatial_map.add_map_to_stats('autocorr', autocorr_OPEXEBO)

    return autocorr_OPEXEBO


def _resize_numpy2D(array: np.ndarray, x: int, y: int) -> np.ndarray:

    '''
        Resizes a numpy array.

        Params:
            array (numpy.ndarray):
                Numpy array to be resized
            x (int):
                Resizing row number (length)
            y (int):
                Resizing column number (width)

        Returns:
            array (numpy.ndarray): Resized array with new dimensions (array.shape = (x,y))
    '''

    array = Image.fromarray(array)
    array = array.resize((x,y))
    array = np.array(array)

    return array
