import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

import numpy as np

def get_binary_map(ratemap: np.ndarray) -> np.ndarray:

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

    binary_map = np.copy(ratemap)
    binary_map[  binary_map >= np.percentile(binary_map.flatten(), 75)  ] = 1
    binary_map[  binary_map < np.percentile(binary_map.flatten(), 75)  ] = 0

    return binary_map


