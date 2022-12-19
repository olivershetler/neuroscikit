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
    obj_output = {}
    keys = ['animal_id','tetrode','unit_id','wass', 'session_ids']
    obj_keys = ['animal_id','tetrode','unit_id','session_id','wass_0', 'wass_90', 'wass_180', 'wass_270', 'wass_no', 'obj_loc', 'obj_x_pos', 'obj_y_pos']

    for key in keys:
        output[key] = []
    for key in obj_keys:
        obj_output[key] = []

    c = 0

    for animal in study.animals:
        
        # get largest possible cell id
        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count))
        remapping_indices = [[] for k in range(max_matched_cell_count)]
        remapping_session_ids = [[] for k in range(max_matched_cell_count)]

        remapping_object_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count, 4))
        remapping_object_indices = [[[] for k in range(4)] for k in range(max_matched_cell_count)]

        # agg_ratemaps = [[] for k in range(len(list(animal.sessions.keys()))-1)]

        for j in range(int(max_matched_cell_count)):
            cell_label = j + 1

            prev = None
            curr = None

            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet'].lower()

                cylinder = check_disk_arena(path)

                ### TEMPORARY WAY TO READ OBJ LOC FROM FILE NAME ###
                if settings['hasObject']:
                    object_location = path.split('/')[-1].split('-')[3].split('.')[0]
                    object_present = True
                    if str(object_location) == 'no':
                        object_present == False
                        object_location = 'no'
                    elif str(object_location) == 'zero':
                        object_location = 0
                    else:
                        object_location = int(object_location)
                        assert int(object_location) in [0,90,180,270]

                if j == 0:
                    assert 'matched' in ses.session_metadata.file_paths['cut'], 'Matched cut file was not used for data loading, cannot proceed with non matched cut file as cluster/cell labels are not aligned'

                pos_obj = ses.get_position_data()['position']

                ensemble = ses.get_cell_data()['cell_ensemble']

                # Check if cell id we're iterating through is present in the ensemble of this sessions
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)

                    spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    rate_map, _ = rate_map_obj.get_rate_map()

                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                    else:
                        curr = rate_map

                    if settings['hasObject']:

                        variations = [0,90,180,270,'no']

                        # compute object remapping for every object position, actual object location is store alongside wass for each object ratemap
                        for var in variations:
                            key = 'wass_' + str(var)

                            object_ratemap, object_pos = make_object_ratemap(var, rate_map_obj)

                            if cylinder:
                                object_ratemap = flat_disk_mask(object_ratemap)

                            # print(object_ratemap.shape)

                            wass, _, _ = compute_wasserstein_distance(object_ratemap, curr)

                            obj_output[key].append(wass)

                        # Store true obj location
                        obj_output['obj_loc'].append(object_location)

                        if object_pos is not None:
                            obj_output['obj_x_pos'].append(object_pos[0])
                            obj_output['obj_y_pos'].append(object_pos[1])
                        else:
                            obj_output['obj_x_pos'].append(None)
                            obj_output['obj_y_pos'].append(None)

                        obj_output['animal_id'].append(animal.animal_id)
                        obj_output['unit_id'].append(cell_label)
                        obj_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        obj_output['session_id'].append(seskey)

                    if prev is not None:
 
                        wass, _, _ = compute_wasserstein_distance(prev, curr)

                        output['animal_id'].append(animal.animal_id)
                        output['unit_id'].append(cell_label)
                        output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        output['session_ids'].append(['session_' + str(i), 'session_' + str(i+1)])
                        output['wass'].append(wass)


                        # point_dist = compute_dist_from_point()

                        # global remapping ?
                        # centroids cdist
                        # nswe 4 direction distance

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

    df = pd.DataFrame(obj_output)
    df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping' + '/obj_remapping.csv')


def make_object_ratemap(object_location, rate_map_obj):
    arena_height, arena_width = rate_map_obj.arena_size
    arena_height = arena_height[0]
    arena_width = arena_width[0]

    rate_map, _ = rate_map_obj.get_rate_map()

    # (64, 64)
    x, y = rate_map.shape

    # convert height/width to arrayswith 64 bins
    height = np.arange(0,arena_height, arena_height/x)
    width = np.arange(0,arena_width, arena_width/y)

    # make zero array same shape as true ratemap == fake ratemap
    arena = np.zeros((len(height),len(width)))

    # if no object, zero across all ratemap
    if object_location == 'no': 
        return arena, [0, 0]

    # if object, pass into dictionary to get x/y coordinates of object location
    object_location_dict = {
        0: [arena_height, arena_width/2],
        90: [arena_height/2, arena_width],
        180: [0, arena_width/2],
        270: [arena_height/2, 0]
    }

    object_pos = object_location_dict[object_location]

    # get x and y ids for the first bin that the object location coordinates fall into
    id_x = np.where(height <= object_pos[0])[0][-1]
    id_y = np.where(width <= object_pos[1])[0][-1]
    # id_x_small = np.where(height < object_pos[0])[0][0]



    # cts_x, _ = np.histogram(object_pos[0], bins=height)
    # cts_y, _ = np.histogram(object_pos[1], bins=width)

    # id_x = np.where(cts_x != 0)[0]
    # id_y = np.where(cts_y != 0)[0]
    # print(arena_height, arena_width, height, width, object_pos, id_x, id_y)

    # set that bin equal to 1
    arena[id_x, id_y] = 1

    return arena, object_pos


def check_disk_arena(path):
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

    return cylinder


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


