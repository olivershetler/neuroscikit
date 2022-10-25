import os, sys
import numpy as np
import matplotlib.pyplot as plt
import ot

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Study
from x_io.rw.axona.batch_read import make_study
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def batch_remapping(paths=[], settings={}, study=None):
    if study is None:
        assert len(paths) > 0 and len(settings) > 0
        # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
        study = make_study(paths, settings)
        # make animals
        study.make_animals()
    elif isinstance(study, Study):
        study.make_animals()

    for animal in study.animals: 

        max_matched_cell_count = len(animal.sessions[sorted(animal.sessions)[-1]].get_cell_data()['cell_ensemble'].cells)

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(animal.sessions)-1, max_matched_cell_count))
        remapping_indices = [[] for i in range(max_matched_cell_count)]
        remapping_session_ids = [[] for i in range(max_matched_cell_count)]

        for j in range(max_matched_cell_count):
            cell_label = j + 1

            
            prev = None 
            curr = None 

            # print('Cell ' + str(cell_label))

            for i in range(len(animal.sessions)):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]

                # print(seskey)
                
                if j == 0:
                    assert 'matched' in ses.session_metadata.file_paths['cut'], 'Matched cut file was not used for data loading, cannot proceed with non matched cut file as cluster/cell labels are not aligned'

                pos_obj = ses.get_position_data()['position']

                ensemble = ses.get_cell_data()['cell_ensemble']
   
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)

                    if 'spatial_spike_train' not in ses.get_spike_data():
                        spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})
                    else:
                        spatial_spike_train = ses.get_spike_data()['spatial_spike_train']

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    rate_map, _ = rate_map_obj.get_rate_map()
                    curr = rate_map

                    if prev is not None:
                        # distance = wasserstein_distance(prev.squeeze(), curr.squeeze())
                        wass = compute_rate_remapping(prev.squeeze(), curr.squeeze())
                       
                        remapping_distances[i-1,cell_label-1] = wass

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                    prev = curr
            
            if 'rate_remapping' not in animal.stats_dict:
                animal.stats_dict['rate_remapping'] = {}
                
            distances = np.asarray(remapping_distances[:, cell_label-1])[remapping_indices[cell_label-1]]

            session_pairs = np.asarray(remapping_session_ids[cell_label-1])

            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)] = {}
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['distances'] = distances
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['session_pairs'] = session_pairs

            print(distances, session_pairs)



def compute_rate_remapping(X, Y):
    # # distance = wasserstein_distance(prev, ses)
    # coords = np.array([X.flatten(), Y.flatten()]).T
    # coordsSqr = np.sum(coords**2, 1)
    # M = coordsSqr[:, None] + coordsSqr[None, :] - 2*coords.dot(coords.T)
    # M[M < 0] = 0
    # M = np.sqrt(M)
    # radius = 0.2
    # I = 1e-5 + np.array((X)**2 + (Y)**2 < radius**2, dtype=float)
    # I /= np.sum(I)

    # l2dist = np.sqrt(np.sum((I)**2))
    # wass = ot.sinkhorn2(I.flatten(), I.flatten(), M, 1.0)
    assert X.shape == Y.shape
    n = X.shape[0]
    d = cdist(X, Y)
    assignment = linear_sum_assignment(d)
    wass = d[assignment].sum() / n

    return wass
    # return wass, l2dist, I


def compute_global_remapping():
    pass