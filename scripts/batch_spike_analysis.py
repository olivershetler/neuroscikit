import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from library.study_space import Study, Animal

from library.cluster import create_features, L_ratio, isolation_distance
from library.spike import histogram_ISI, find_burst

from PIL import Image
import numpy as np
from matplotlib import cm

def batch_spike_analysis(study: Study):
    """
    Computes rate maps across all animals, sessions, cells in a study.

    Use tasks dictionary as true/false flag with variable to compute
    e.g. {'rate_map': True, 'binary_map': False}
    """

    study.make_animals()
    animals = study.animals

    for animal in animals:

        c = 1
        for session_key in animal.sessions:
            session = animal.sessions[session_key]

            k = 1

            cluster_batch = session.get_spike_data()['spike_cluster']
            create_features(cluster_batch)
            L_ratio(cluster_batch)
            isolation_distance(cluster_batch)

            for cell in session.get_cell_data()['cell_ensemble'].cells[:1]:

                print('session ' + str(c) + ', cell ' + str(k))

                find_burst(cell)
                histogram_ISI(cell)

                k += 1

            c += 1





