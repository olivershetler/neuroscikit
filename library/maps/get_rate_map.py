import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

import numpy as np
import cv2
from library.maps.get_occupancy_map import get_occupancy_map
from library.maps.get_spike_map import get_spike_map

def get_rate_map(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray,
                arena_size: tuple, spikex: np.ndarray, spikey: np.ndarray,
                kernlen: int, std: int) -> np.ndarray:

    '''
        Computes a 2D array encoding neuron spike events in 2D space, based on where
        the subject walked during experiment.

        Params:
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y coordinates and timestamps
            arena_size (tuple):
                Arena dimensions based on x,y coordinates
            spikex, spikey (np.ndarray's):
                x and y coordinates of spike occurence
            kernlen, std (int):
                kernel size and std for convolutional smoothing

        Returns:
            Tuple: (x_resize, y__resize)
                --------
                ratemap_smoothed (np.ndarray):
                    Smoothed ratemap
                ratemap_raw (np.ndarray):
                    Raw ratemap
    '''

    # import pdb; pdb.set_trace()
    occ_map_smoothed, occ_map_raw, _ = get_occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

    spike_map_smoothed, spike_map_raw = get_spike_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)

    # Compute ratemap
    ratemap_raw = np.where(occ_map_raw<0.0001, 0, spike_map_raw/occ_map_raw)
    ratemap_smooth = np.where(occ_map_smoothed<0.0001, 0, spike_map_smoothed/occ_map_smoothed)
    ratemap_smooth = ratemap_smooth/max(ratemap_smooth.flatten())

    # Smooth ratemap
    #ratemap_a_smooth = _interpolate_matrix(ratemap_raw)

    #ratemap_b_smooth = np.where(occ_map_smoothed<0.00001, 0, spikemap_smoothed/occ_map_smoothed)
    #ratemap_b_smooth = ratemap_b_smooth / max(ratemap_b_smooth.flatten())

    return ratemap_smooth, ratemap_raw
