import os, sys
import numpy as np
import matplotlib.pyplot as plt
import ot
import pandas as pd
import regex as re

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Study
from x_io.rw.axona.batch_read import make_study
from library.hafting_spatial_maps import HaftingRateMap, SpatialSpikeTrain2D
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image
from matplotlib import cm
from library.maps.map_utils import _interpolate_matrix, disk_mask
import cv2
import openpyxl as xl
from openpyxl.utils.cell import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension


def batch_remapping(paths=[], settings={}, study=None):
    if study is None:
        assert len(paths) > 0 and len(settings) > 0
        # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
        study = make_study(paths, settings)
        # make animals
        study.make_animals()
    elif isinstance(study, Study):
        study.make_animals()

    output = {}
    keys = ['animal_id','tetrode','unit_id','wasserstein', 'session_ids']

    for key in keys:
        output[key] = []

    c = 0

    for animal in study.animals:

        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count))
        remapping_indices = [[] for k in range(max_matched_cell_count)]
        remapping_session_ids = [[] for k in range(max_matched_cell_count)]

        # agg_ratemaps = [[] for k in range(len(list(animal.sessions.keys()))-1)]

        for j in range(int(max_matched_cell_count)):
            cell_label = j + 1

            prev = None
            curr = None

            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet'].lower()

                variations = [r'cylinder', r'round', r'circle']
                var_bool = []
                for var in variations:
                    if re.search(var, path) is not None:
                        var_bool.append(True)
                    else:
                        var_bool.append(False)
                # if re.search(r'cylinder', path) is not None or re.search(r'round', path) is not None:

                if np.array(var_bool).any() == True:
                    cylinder = True
                else:
                    cylinder = False

                if j == 0:
                    assert 'matched' in ses.session_metadata.file_paths['cut'], 'Matched cut file was not used for data loading, cannot proceed with non matched cut file as cluster/cell labels are not aligned'

                pos_obj = ses.get_position_data()['position']

                ensemble = ses.get_cell_data()['cell_ensemble']

                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)
                    # print(cell, print(cell.event_times))

                    # if 'spatial_spike_train' not in ses.get_spike_data():
                    #     spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})
                    # else:
                    #     spatial_spike_train = ses.get_spike_data()['spatial_spike_train']

                    spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    rate_map, _ = rate_map_obj.get_rate_map()
                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                    else:
                        curr = rate_map

                    if prev is not None:
                        #if cylender:
                        #    prev = flat_disk_mask(prev)
                        wass, _, _ = compute_wasserstein_distance(prev, curr)

                        output['animal_id'].append(animal.animal_id)
                        output['unit_id'].append(cell_label)
                        output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        output['session_ids'].append(['session_' + str(i), 'session_' + str(i+1)])
                        output['wasserstein'].append(wass)

                        # point_dist = compute_dist_from_point()

                        # global remapping ?
                        # centroids cdist
                        # nswe 4 direction distance
                        #
                        # print(wass)
                        # print(prev[:10], curr[:10], wass)
                        remapping_distances[i-1,cell_label-1] = wass

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                        # toplot = _interpolate_matrix(prev, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
                        # colored_ratemap = Image.fromarray(np.uint8(cm.jet(toplot)*255))

                        # colored_ratemap.save('ratemap_cell_' + str(c) + '.png')
                        c += 1
                    else:
                        prev = np.copy(curr)

            if 'rate_remapping' not in animal.stats_dict:
                animal.stats_dict['rate_remapping'] = {}

            distances = np.asarray(remapping_distances[:, cell_label-1])[remapping_indices[cell_label-1]]

            session_pairs = np.asarray(remapping_session_ids[cell_label-1])

            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)] = {}
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['distances'] = distances
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['session_pairs'] = session_pairs

            # toplot = _interpolate_matrix(curr, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
            # colored_ratemap = Image.fromarray(np.uint8(cm.jet(toplot)*255))
            # print(i, j)
            # colored_ratemap.save('ratemap_cell_' + str(c) + '.png')
            c += 1

        print(remapping_distances, remapping_session_ids)

        # agg_session_wass = compute_global_remapping(agg_ratemaps, animal)

    df = pd.DataFrame(output)
    df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping' + '/rate_remapping.csv')


# def compute_global_remapping(agg_ratemaps, animal):
#     prevAvg = None
#     currAvg = None
#     agg_session_wass = {}

#     keys = ['animal_id','wasserstein', 'session_ids']

#     for key in keys:
#         agg_session_wass[key] = []

#     for k in range(len(agg_ratemaps)):
#         avg_ratemap = avg_session_ratemaps(agg_ratemaps[k])
#         currAvg = avg_ratemap
#         if prevAvg is not None:
#             session_wass = compute_wasserstein_distance(prevAvg, currAvg)
#             agg_session_wass['animal_id'] = animal.animal_id
#             agg_session_wass['wasserstein'] = session_wass
#             agg_session_wass['session_ids'] = ['session_' + str(k+1),'session_' + str(k+2)]
#         prevAvg = currAvg
#     return agg_session_wass

def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = 0
    return  masked_rate_map.data

def compute_dist_from_point(xcoord, ycoord):
    pass

def compute_wasserstein_distance(X, Y):
    # distance = wasserstein_distance(prev, ses)
    coords = np.array([X.flatten(), Y.flatten()]).T
    coordsSqr = np.sum(coords**2, 1)
    M = coordsSqr[:, None] + coordsSqr[None, :] - 2*coords.dot(coords.T)
    M[M < 0] = 0
    M = np.sqrt(M)
    radius = 0.2
    I = 1e-5 + np.array((X)**2 + (Y)**2 < radius**2, dtype=float)
    I /= np.sum(I)

    l2dist = np.sqrt(np.sum((I)**2))
    wass = ot.sinkhorn2(I.flatten(), I.flatten(), M, 1.0)

    # assert X.shape == Y.shape
    # n = X.shape[0]
    # d = cdist(X, Y)
    # assignment = linear_sum_assignment(d)
    # wass = d[assignment].sum() / n
    # return wass

    return wass, l2dist, I

# def avg_session_ratemaps(ratemaps):
#     avg_ratemap = []

#     for i in range(len(ratemaps)):
#         if i == 0:
#             avg_ratemap = ratemaps[i]
#         else:
#             avg_ratemap = avg_ratemap + ratemaps[i]

#     avg_ratemap = avg_ratemap / len(ratemaps)

#     return avg_ratemap

