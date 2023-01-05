import os, sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.settings import rate_output, obj_output, centroid_output
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_rate_remapping
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, single_point_wasserstein, pot_sliced_wasserstein, compute_centroid_remapping
from _prototypes.cell_remapping.src.masks import make_object_ratemap, check_disk_arena, flat_disk_mask
from library.maps import map_blobs

import matplotlib.pyplot as plt

"""

TODO (in order of priority)

- Pull out POT dependecies for sliced wass - DONE
- take all spikes in cell, get (x,y) position, make sure they are scaled properly (i.e. in cm/inches) at file loading part - DONE
- read ppm from position file ('pixels_per_meter' word search) and NOT settings.py file - DONE
- check ppm can bet set at file loading, you likely gave that option if 'ppm' not present in settings - DONE

- Use map blobs to get fields - DONE
- get idx in fields and calculate euclidean distance for all permutations of possible field combinations - DONE
- can start with only highest density fields - DONE
- refactor identified + appropriate areas into helper functions (especially map blobs related code) to simplify 
- MUST revisit map blobs and how the 90th percentile is being done 
- Reconcile definition of fields with papers Abid shared in #code to make field definition for our case concrete
- visualize selected fields (plot ratemap + circle/highlight in diff color idx of each field, can plot binary + ratemap below to show true density in field)

- Implement rotation remapping, get ready for case where from session to session field map is rotated by 0/90/180 etc instead of object location
- Will have to do the same as object case where you do every rotation permutation and store the true rotation angle to look at wass distances 

- Implement globabl remapping? Just average ratemaps across all cells in session and use average ratemap of each sessionn in sliced wass

"""


def compute_remapping(study, settings):

    c = 0
    
    max_centroid_count, blobs_dict = _find_largest_centroid_count(study)

    for animal in study.animals:

        # get largest possible cell id
        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count))
        remapping_indices = [[] for k in range(max_matched_cell_count)]
        remapping_session_ids = [[] for k in range(max_matched_cell_count)]

        # for every existing cell id across all sessions
        for j in range(int(max_matched_cell_count)):
            cell_label = j + 1

            prev = None
            curr = None
            # max_centroid_count = None

            # for every session
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet'].lower()

                # Check if cylinder
                cylinder, true_var = check_disk_arena(path)
                ###### TEMPORARILY FORCING TO TRUE PLEASE REMOVE THIS AFTER DONE TESTING
                # cylinder = True

                ### TEMPORARY WAY TO READ OBJ LOC FROM FILE NAME ###
                if settings['hasObject']:
                    object_location = _read_location_from_file(path, cylinder, true_var)

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
                    
                    # Disk mask ratemap
                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                    else:
                        curr = rate_map
                    
                    curr_spatial_spike_train = spatial_spike_train

                    if settings['hasObject']:

                        variations = [0,90,180,270,'no']

                        # compute object remapping for every object position, actual object location is store alongside wass for each object ratemap
                        for var in variations:
                            obj_wass_key = 'obj_wass_' + str(var)

                            object_ratemap, object_pos = make_object_ratemap(var, rate_map_obj)

                            if var == object_location:
                                true_object_pos = object_pos
                                true_object_ratemap = object_ratemap
                            
                            # disk mask fake object ratemap
                            ###### TESTING THIS,MAYBE DONT NEED TO DISK MASK THE ARTIFICAL OBJECT RATEMAP
                            if cylinder:
                                object_ratemap = flat_disk_mask(object_ratemap)

                            rate_map, _ = rate_map_obj.get_rate_map()

                            obj_wass = single_point_wasserstein(object_pos, rate_map, rate_map_obj.arena_size)

                            obj_output[obj_wass_key].append(obj_wass)

                        # Store true obj location
                        obj_output['object_location'].append(object_location)

                        # if object_pos is not None:
                        obj_output['obj_pos_x'].append(true_object_pos['x'])
                        obj_output['obj_pos_y'].append(true_object_pos['y'])

                        obj_output['animal_id'].append(animal.animal_id)
                        obj_output['unit_id'].append(cell_label)
                        obj_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        obj_output['session_id'].append(seskey)

                        plot_obj_remapping(true_object_ratemap, curr, obj_output)

                    curr_id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)

                    if prev is not None:
                        # num_proj = 100
                        # wass, _, _ = compute_wasserstein_distance(prev, curr)
                        # sliced_wass = sliced_wasserstein(prev, curr, num_proj)
                        prev_spike_pos_x, prev_spike_pos_y, _ = prev_spatial_spike_train.get_spike_positions()
                        prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).reshape((-1,2))

                        curr_spike_pos_x, curr_spike_pos_y, _ = curr_spatial_spike_train.get_spike_positions()
                        curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).reshape((-1,2))

                        sliced_wass = pot_sliced_wasserstein(prev_pts, curr_pts)

                        rate_output['animal_id'].append(animal.animal_id)
                        rate_output['unit_id'].append(cell_label)
                        rate_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        rate_output['session_ids'].append(['session_' + str(i), 'session_' + str(i+1)])
                        # rate_output['wass'].append(wass)
                        rate_output['sliced_wass'].append(sliced_wass)

                        plot_rate_remapping(prev, curr, rate_output)

                        
                        # image_prev, n_labels_prev, labels_prev, centroids_prev, field_sizes_prev = map_blobs(prev_spatial_spike_train)

                        # image_curr, n_labels_curr, labels_curr, centroids_curr, field_sizes_curr = map_blobs(curr_spatial_spike_train)

                        image_prev, n_labels_prev, labels_prev, centroids_prev, field_sizes_prev = blobs_dict[prev_id]

                        image_curr, n_labels_curr, labels_curr, centroids_curr, field_sizes_curr = blobs_dict[curr_id]

                        sort_idx_source = np.argsort(-np.array(field_sizes_prev))
                        source_labels = np.zeros(labels_prev.shape)

                        map_dict = {}
                        for k in np.unique(labels_prev):
                            if k != 0:
                                row, col = np.where(labels_prev == k)
                                source_labels[row, col] = sort_idx_source[k-1] + 1
                                map_dict[k] = sort_idx_source[k-1] + 1
                            else:
                                map_dict[k] = 0

                        out = np.vectorize(map_dict.get)(labels_prev)

                        assert out.all() == source_labels.all()

                        # print(map_dict, np.unique(source_labels), np.unique(labels_prev))

                        # fig = plt.figure(figsize=(6,6))
                        # plt.imshow(source_labels)
                        # plt.show()

                        # fig = plt.figure(figsize=(6,6))
                        # plt.imshow(labels_prev)
                        # plt.show()

                        assert len(np.unique(source_labels)) == len(np.unique(labels_prev))

                        sort_idx_target = np.argsort(-np.array(field_sizes_curr))

                        centroid_wass, centroid_pairs = compute_centroid_remapping(centroids_curr[sort_idx_target], source_labels, prev_spatial_spike_train)

                        # print(len(centroid_wass), len(field_sizes_prev)*len(field_sizes_curr))

                        assert len(centroid_wass) == len(field_sizes_prev) * len(field_sizes_curr)

                        # if max_centroid_count > len(centroid_wass):
                        #     rng = max_centroid_count
                        # else:
                        #     rng = len(centroid_wass)

                        prev_key_len = None
                        # prev_wass_key = None 
                        # prev_id_key = None

                        # max_key_len = 0

                        centroid_output['animal_id'].append(animal.animal_id)
                        centroid_output['unit_id'].append(cell_label)
                        centroid_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        centroid_output['session_ids'].append(['session_' + str(i), 'session_' + str(i+1)])
                        # centroid_output['cumulative_wass'].append()

                        # print('rng: ' + str(max_centroid_count) + ' prev: ' + str(max_centroid_count) + ' curr: ' + str(len(centroid_wass)))
                        for n in range(max_centroid_count):
                            wass_key = 'centroid_wass_'+str(n+1)
                            id_key = 'centroid_ids_'+str(n+1)

                            if wass_key not in centroid_output:
                                centroid_output[wass_key] = []
                                centroid_output[id_key] = []
                            # else: 

                            if n < len(centroid_wass):
                                centroid_output[wass_key].append(centroid_wass[n])
                                centroid_output[id_key].append(centroid_pairs[n])
                            else:
                                centroid_output[wass_key].append(0)
                                centroid_output[id_key].append([0,0])

                            # curr_key_len = len(rate_output[wass_key])

                            if prev_key_len is not None and prev_key_len > len(centroid_output[wass_key]):
                                diff = prev_key_len - len(centroid_output[wass_key])

                                print('Should not see this line')

                                assert diff > 0
                                # print(diff, len(rate_output[wass_key]))
                                for k in range(diff):
                                    centroid_output[wass_key] = [0] + centroid_output[wass_key]
                                    centroid_output[id_key] = [[0,0]] + centroid_output[id_key]

                                # print(prev_key_len,  len(rate_output[wass_key]))
                                assert prev_key_len == len(centroid_output[wass_key])

                            prev_key_len = len(centroid_output[wass_key])
                            # prev_wass_key = wass_key
                            # prev_id_key = id_key

                            # if curr_key_len > max_key_len:
                            #     max_key_len = curr_key_len

                        # if 'centroid_comp_count' not in rate_output:
                        #     rate_output['centroid_comp_count'] = []

                        # rate_output['centroid_comp_count'].append(rng)

                        # global remapping ?
                        # centroids cdist
                        # nswe 4 direction distance

                        # remapping_distances[i-1,cell_label-1] = wass

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                        # if max_centroid_count is not None:
                        #     max_centroid_count = max(max_centroid_count, len(centroid_wass))
                        # else:
                        #     max_centroid_count = len(centroid_wass)

                        c += 1
                    else:
                        prev = np.copy(curr)
                        prev_spatial_spike_train = curr_spatial_spike_train
                        prev_id = curr_id

            # if 'rate_remapping' not in animal.stats_dict:
            #     animal.stats_dict['rate_remapping'] = {}

            # prev_max_centroid_count = max_centroid_count

            # distances = np.asarray(remapping_distances[:, cell_label-1])[remapping_indices[cell_label-1]]

            # session_pairs = np.asarray(remapping_session_ids[cell_label-1])

            # animal.stats_dict['rate_remapping']['cell_' + str(cell_label)] = {}
            # animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['distances'] = distances
            # animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['session_pairs'] = session_pairs

            c += 1

    return {'rate': rate_output, 'object': obj_output, 'centroid': centroid_output}



def _read_location_from_file(path, cylinder, true_var):

    if not cylinder:
        object_location = path.split('/')[-1].split('-')[3].split('.')[0]
    else:
        items = path.split('/')[-1].split('-')
        idx = items.index(str(true_var)) + 2 # the object location is always 2 positions away from word denoting arena hape (e.g round/cylinder) defined by true_var
        # e.g. ROUND-3050-90_2.clu
        object_location = items[idx].split('.')[0].split('_')[0]

    object_present = True
    if str(object_location) == 'no':
        object_present == False
        object_location = 'no'
    elif str(object_location) == 'zero':
        object_location = 0
    else:
        object_location = int(object_location)
        assert int(object_location) in [0,90,180,270]

    return object_location

def _find_largest_centroid_count(study):
    max_centroid_count = 0
    blobs_dict = {}
    for animal in study.animals:
        for i in range(len(list(animal.sessions.keys()))):
            seskey = 'session_' + str(i+1)
            ses = animal.sessions[seskey]
            for cell in ses.get_cell_data()['cell_ensemble'].cells:
                spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': ses.get_position_data()['position']})

                image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train)

                id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)

                blobs_dict[id] = [image, n_labels, labels, centroids, field_sizes]

                print(len(field_sizes))
                max_centroid_count = max(max_centroid_count, len(field_sizes))