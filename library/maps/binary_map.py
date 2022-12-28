import os
import sys

from library import spatial

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


import numpy as np
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
# from library.spatial_spike_train import SpatialSpikeTrain2D


def binary_map(spatial_map: HaftingRateMap | SpatialSpikeTrain2D, percentile=.75 **kwargs):

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
        smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

    if isinstance(spatial_map, HaftingRateMap):
        ratemap, _ = spatial_map.get_rate_map(smoothing_factor)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        ratemap, _ = spatial_map.get_map('rate').get_rate_map(smoothing_factor)

    binary_map = np.zeros(ratemap.shape)
    binary_map[  binary_map >= np.percentile(binary_map.flatten(), percentile)  ] = 1

    if isinstance(spatial_map, HaftingRateMap):
        spatial_map.spatial_spike_train.add_map_to_stats('binary', binary_map)
    elif isinstance(spatial_map, SpatialSpikeTrain2D):
        spatial_map.add_map_to_stats('binary', binary_map)

    return binary_map


