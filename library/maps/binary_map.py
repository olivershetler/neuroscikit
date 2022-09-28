import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

import numpy as np
from library.hafting_spatial_maps import HaftingRateMap
from library.spatial_spike_train import SpatialSpikeTrain2D


def binary_map(spatial_map: HaftingRateMap | SpatialSpikeTrain2D, **kwargs):

    '''
        Computes a binary map of the ratemap where only areas of moderate to
            high ratemap activity are captured.

        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            np.ndarray:
                binary_map
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

    binary_map = np.copy(ratemap)
    binary_map[  binary_map >= np.percentile(binary_map.flatten(), 75)  ] = 1
    binary_map[  binary_map < np.percentile(binary_map.flatten(), 75)  ] = 0

    spatial_map.stats_dict['binary'] = binary_map

    return binary_map


