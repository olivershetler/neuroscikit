import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

import numpy as np
import cv2
from library.maps.map_utils import _compute_resize_ratio, _interpolate_matrix, _gkern

def get_occupancy_map(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray, arena_size: tuple, kernlen: int, std: int) -> np.ndarray:

    '''
        Computes the position, or occupancy map, which is a 2D numpy array
        enconding subjects position over entire experiment.

        Params:
            pos_x, pos_y and pos_t (np.ndarray):
                Arrays of the subjects x and y coordinates, as
                well as timestamp array.
            arena_size (tuple):
                Arena dimensions (width (x), height (y) in meters)
            resolution (float):
                Resolution of occupancy map (in meters)
            kernlen, std : kernel size and standard deviation (i.e 'spread') for convolutional smoothing
                of 2D map

            Returns:
                np.ndarray: occ_map_smoothed, occ_map_raw, coverage_map
    '''


    min_x = min(pos_x)
    max_x = max(pos_x)
    min_y = min(pos_y)
    max_y = max(pos_y)

    arena_size = (abs(max_y-min_y), abs(max_x - min_x)) # (height, width)

    # Resize ratio
    row_resize, column_resize = _compute_resize_ratio(arena_size)

    # Initialize empty map
    occ_map_raw = np.zeros((row_resize,column_resize))
    coverage_map = np.zeros((row_resize,column_resize))
    row_values = np.linspace(max_y,min_y,row_resize)
    column_values = np.linspace(min_x,max_x,column_resize)

    # Generate the raw occupancy map
    for i in range(1, len(pos_t)):

        row_index = np.abs(row_values - pos_y[i]).argmin()
        column_index = np.abs(column_values - pos_x[i]).argmin()
        occ_map_raw[row_index][column_index] += pos_t[i] - pos_t[i-1]
        coverage_map[row_index][column_index] = 1

    # Normalize and smooth with scaling facotr
    occ_map_normalized = occ_map_raw / pos_t[-1]
    occ_map_smoothed = cv2.filter2D(occ_map_normalized,-1,_gkern(kernlen,std))

    # dilate coverage map
    kernel = np.ones((2,2))
    coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)

    # Resize maps
    occ_map_raw = _interpolate_matrix(occ_map_raw, cv2_interpolation_method=cv2.INTER_NEAREST)
    occ_map_smoothed = _interpolate_matrix(occ_map_smoothed, cv2_interpolation_method=cv2.INTER_NEAREST)
    occ_map_smoothed = occ_map_smoothed/max(occ_map_smoothed.flatten())

    return occ_map_smoothed, occ_map_raw, coverage_map
