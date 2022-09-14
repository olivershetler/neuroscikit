import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.maps import feature_wave_PCX, feature_energy

import numpy as np

def create_features(data, featuresToCalculate=['energy', 'wave_PCX!1']):
    """Creates the features to be analyzed

    Args:
        featuresToCalculate:

    Returns:
        FD:

    """

    FD = np.array([])

    for feature_name in featuresToCalculate:
        if 'wave_PCX' in feature_name:
            # instead of creating a function for every PCX you want to use, I just
            # created one. Use WavePCX!1 for PC1, WavePCX!2 for PC2, etc.

            # all the feature functions are named 'feature_featureName'
            # WavePCX is special though since you need a PC number so separate that
            feature_name, pc_number = feature_name.split('!')
            variables = 'data, %s' % pc_number
        else:
            variables = 'data'

        fnc_name = 'feature_%s' % feature_name
        current_FD = eval("%s(%s)" % (fnc_name, variables))

        if len(FD) == 0:
            FD = current_FD
        else:
            FD = np.hstack((FD, current_FD))

        current_FD = None

    return FD