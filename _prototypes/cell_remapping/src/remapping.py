import os, sys
import numpy as np
import itertools

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_rate_remapping
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, single_point_wasserstein, pot_sliced_wasserstein, compute_centroid_remapping, _get_ratemap_bucket_midpoints
from _prototypes.cell_remapping.src.masks import make_object_ratemap, check_disk_arena, flat_disk_mask
from library.maps import map_blobs
from scripts.batch_map.batch_map import batch_map
from _prototypes.cell_remapping.src.settings import obj_output, centroid_output, tasks, session_comp_categories, rate_output, context_output, variations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


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
- refactor identified + appropriate areas into helper functions (especially map blobs related code) to simplify - DONE

- add comments 
- MUST revisit map blobs and how the 90th percentile is being done 
- Reconcile definition of fields with papers Abid shared in #code to make field definition for our case concrete
- visualize selected fields (plot ratemap + circle/highlight in diff color idx of each field, can plot binary + ratemap below to show true density in field)

- Implement rotation remapping, get ready for case where from session to session field map is rotated by 0/90/180 etc instead of object location
- Will have to do the same as object case where you do every rotation permutation and store the true rotation angle to look at wass distances 

- Implement globabl remapping? Just average ratemaps across all cells in session and use average ratemap of each session in sliced wass

"""


def compute_remapping(study, settings):

    c = 0

    batch_map(study, tasks)
    
    max_centroid_count, blobs_dict = _aggregate_cell_info(study)

    centroid_dict = centroid_output
    rate_dict = rate_output
    context_dict = context_output
    obj_dict = obj_output

    for animal in study.animals:

        # get largest possible cell id
        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count))
        remapping_indices = [[] for k in range(max_matched_cell_count)]
        remapping_session_ids = [[] for k in range(max_matched_cell_count)]

        # for every existing cell id across all sessions
        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1

            # prev ratemap
            prev = None
            # comp_prev = None

            # for every session
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet'].lower()

                # Check if cylinder
                cylinder, true_var = check_disk_arena(path)

                ### TEMPORARY WAY TO READ OBJ LOC FROM FILE NAME ###
                if settings['hasObject']:
                    object_location = _read_location_from_file(path, cylinder, true_var)

                ensemble = ses.get_cell_data()['cell_ensemble']

                # Check if cell id we're iterating through is present in the ensemble of this sessions
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)

                    # spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})

                    spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    rate_map, _ = rate_map_obj.get_rate_map()
                    
                    # Disk mask ratemap
                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                    else:
                        curr = rate_map
                    
                    curr_cell = cell
                    curr_spatial_spike_train = spatial_spike_train
                    curr_key = seskey

                    # If object used in experiment 
                    if settings['hasObject']:
                        
                        # # Possible object locations (change to get from settings)
                        # variations = [0,90,180,270,'no']

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

                            obj_dict[obj_wass_key].append(obj_wass)

                        # Store true obj location
                        obj_dict['object_location'].append(object_location)

                        # if object_pos is not None:
                        obj_dict['obj_pos_x'].append(true_object_pos['x'])
                        obj_dict['obj_pos_y'].append(true_object_pos['y'])

                        obj_dict['animal_id'].append(animal.animal_id)
                        obj_dict['unit_id'].append(cell_label)
                        obj_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                        obj_dict['session_id'].append(seskey)

                        if settings['plotObject']:
                            plot_obj_remapping(true_object_ratemap, curr, obj_dict)

                    curr_id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)

                    # If prev ratemap is not None (= we are at session2 or later, session1 has no prev session to compare)
                    if prev is not None:

                        # get x and y pts for spikes in pair of sessions (prev and curr) for a given comparison

                        prev_spike_pos_x, prev_spike_pos_y, _ = prev_spatial_spike_train.get_spike_positions()
                        # prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).reshape((-1,2))
                        prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).T

                        curr_spike_pos_x, curr_spike_pos_y, _ = curr_spatial_spike_train.get_spike_positions()
                        # curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).reshape((-1,2))
                        curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).T

                        # Get cts from spike rate on ratemap
                        prev_pts_ct = np.array(prev * 1000, dtype=np.int32)
                        curr_pts_ct = np.array(curr * 1000, dtype=np.int32)

                        height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(prev_spatial_spike_train.arena_size, 64, 64)

                        # Use cts to 'weigh' number of (x,y) pts contributed by each spike
                        prev_coords = list(map(lambda i,j: [[height_bucket_midpoints[i], width_bucket_midpoints[j]] for x in range(int(prev_pts_ct[i,j]))], np.arange(0,prev_pts_ct.shape[0],1), np.arange(0,prev_pts_ct.shape[1],1)))
                        curr_coords = list(map(lambda i,j: [[height_bucket_midpoints[i], width_bucket_midpoints[j]] for x in range(int(curr_pts_ct[i,j]))], np.arange(0,curr_pts_ct.shape[0],1), np.arange(0,curr_pts_ct.shape[1],1)))

                        prev_coords = list(itertools.chain.from_iterable(prev_coords))
                        curr_coords = list(itertools.chain.from_iterable(curr_coords))

                        # Test wass is adjusted spike pos
                        test_wass = pot_sliced_wasserstein(prev_coords, curr_coords)
                        # Sliced wass is without adjustemennt
                        sliced_wass = pot_sliced_wasserstein(prev_pts, curr_pts)
                        
                        rate_dict['animal_id'].append(animal.animal_id)
                        rate_dict['unit_id'].append(cell_label)
                        rate_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                        rate_dict['session_ids'].append([prev_key, curr_key])
                        rate_dict['sliced_wass'].append(sliced_wass)
                        rate_dict['test_wass'].append(test_wass)

                        # d = cdist(prev_pts, curr_pts)
                        # assignment = linear_sum_assignment(d)
                        # test_wass = d[assignment].sum() / spatial_spike_train.arena_size[0]
                        # rate_dict['test_wass'].append(test_wass)

                        # rate_dict = _fill_cell_type_stats(rate_dict, prev_cell, curr_cell)

                        # rate_dict['information'].append([prev_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content']-prev_cell.stats_dict['cell_stats']['spatial_information_content']])
                        # rate_dict['grid_score'].append([prev_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score']-prev_cell.stats_dict['cell_stats']['grid_score']])
                        # rate_dict['b_top'].append([prev_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top']-prev_cell.stats_dict['cell_stats']['b_score_top']])
                        # rate_dict['b_bottom'].append([prev_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom']-prev_cell.stats_dict['cell_stats']['b_score_bottom']])
                        # rate_dict['b_right'].append([prev_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right']-prev_cell.stats_dict['cell_stats']['b_score_right']])
                        # rate_dict['b_left'].append([prev_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left']-prev_cell.stats_dict['cell_stats']['b_score_left']])
                        
                        # rate_dict['speed_score'].append([prev_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score']-prev_cell.stats_dict['cell_stats']['speed_score']])
                        # rate_dict['hd_score'].append([prev_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score']-prev_cell.stats_dict['cell_stats']['hd_score']])


                        if settings['plotRate']:
                            plot_rate_remapping(prev, curr, rate_dict)

                        if settings['runFields']:
                            # blobs_dict_curr = curr_cell.stats['field_size_data']
                            # blobs_dict_prev = prev_cell.stats['field_size_data']

                            image_prev, n_labels_prev, labels_prev, centroids_prev, field_sizes_prev = blobs_dict[prev_id]

                            image_curr, n_labels_curr, labels_curr, centroids_curr, field_sizes_curr = blobs_dict[curr_id]

                            target_labels, source_labels, source_centroids, target_centroids = _sort_centroids_by_field_size(field_sizes_prev, field_sizes_curr, labels_prev, labels_curr, centroids_prev, centroids_curr)

                            # prev spatial spike train is source spatial spike train
                            field_wass, field_pairs, cumulative_wass, test_wass, centroid_wass, binary_wass = compute_centroid_remapping(target_labels, source_labels, curr_spatial_spike_train, prev_spatial_spike_train, target_centroids, source_centroids)
                            # cumulative_wass = compute_cumulative_centroid_remapping(target_centers, source_labels, prev_spatial_spike_train, field_sizes_curr)

                            centroid_dict['animal_id'].append(animal.animal_id)
                            centroid_dict['unit_id'].append(cell_label)
                            centroid_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                            centroid_dict['session_ids'].append([prev_key, curr_key])
                            centroid_dict['cumulative_wass'].append([cumulative_wass, np.sum(centroid_wass), np.sum(binary_wass)])
                            # centroid_dict['binary_wass'].append(binary_wass)

                            wass_args = [field_wass, centroid_wass, binary_wass]

                            # need to make test_wass save as in fill centroid dict
                            # centroid_dict['test_wass'].append(test_wass)
                            # centroid_dict['centroid_wass'].append(centroid_wass)

                            centroid_dict = _fill_centroid_dict(centroid_dict, max_centroid_count, wass_args, field_pairs)

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                        c += 1

                    prev = np.copy(curr)
                    prev_spatial_spike_train = curr_spatial_spike_train
                    prev_id = curr_id
                    prev_cell = curr_cell
                    prev_key = curr_key
            
            # If there are context specific or otherwise specific groups to compare, can set those ids in settings
            # Will perform session to session remapping segregated by groups definned in settings
            if settings['runUniqueGroups']:
                # for category group (group of session ids)
                for categ in session_comp_categories:
                    comp_categories = session_comp_categories[categ]
                    prev_key = None 
                    comp_prev = None 
                    comp_prev_cell = None

                    # For session in that category
                    for comp_ses in comp_categories:
                        seskey = 'session_' + str(comp_ses)
                        ses = animal.sessions[seskey]
                        path = ses.session_metadata.file_paths['tet'].lower()
                        cylinder, true_var = check_disk_arena(path)

                        ensemble = ses.get_cell_data()['cell_ensemble']

                        if cell_label in ensemble.get_cell_label_dict():
                            cell = ensemble.get_cell_by_id(cell_label)

                            spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']
                            # rate_map_obj = spatial_spike_train.get_map('rate')
                            # rate_map, _ = rate_map_obj.get_rate_map()

                            comp_curr = spatial_spike_train
                            comp_curr_cell = cell
                            curr_key = seskey

                            if comp_prev is not None:
                                prev_spike_pos_x, prev_spike_pos_y, _ = comp_prev.get_spike_positions()
                                # prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).reshape((-1,2))
                                prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).T

                                curr_spike_pos_x, curr_spike_pos_y, _ = comp_curr.get_spike_positions()
                                # curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).reshape((-1,2))
                                curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).T

                                sliced_wass = pot_sliced_wasserstein(prev_pts, curr_pts)

                                context_dict[categ]['animal_id'].append(animal.animal_id)
                                context_dict[categ]['unit_id'].append(cell_label)
                                context_dict[categ]['tetrode'].append(animal.animal_id.split('tet')[-1])
                                context_dict[categ]['session_ids'].append([prev_key, curr_key])
                                context_dict[categ]['sliced_wass'].append(sliced_wass)

                                # context_dict[categ] = _fill_cell_type_stats(context_dict[categ], comp_prev_cell, comp_curr_cell)

                                # context_dict['information'].append([prev_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['ratemap_stats_dict']['spatial_information_content']-prev_cell.stats_dict['cell_stats']['spatial_information_content']])
                                # context_dict['grid_score'].append([prev_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score']-prev_cell.stats_dict['cell_stats']['grid_score']])
                                # context_dict['b_top'].append([prev_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top']-prev_cell.stats_dict['cell_stats']['b_score_top']])
                                # context_dict['b_bottom'].append([prev_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom']-prev_cell.stats_dict['cell_stats']['b_score_bottom']])
                                # context_dict['b_right'].append([prev_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right']-prev_cell.stats_dict['cell_stats']['b_score_right']])
                                # context_dict['b_left'].append([prev_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left']-prev_cell.stats_dict['cell_stats']['b_score_left']])
                                
                                # context_dict['speed_score'].append([prev_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score']-prev_cell.stats_dict['cell_stats']['speed_score']])
                                # context_dict['hd_score'].append([prev_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score']-prev_cell.stats_dict['cell_stats']['hd_score']])

                            comp_prev = comp_curr
                            comp_prev_cell = comp_curr_cell
                            prev_key = curr_key

            c += 1

    return {'rate': rate_dict, 'object': obj_dict, 'centroid': centroid_dict, 'context': context_dict}

def _fill_cell_type_stats(inp_dict, prev_cell, curr_cell):
    inp_dict['information'].append([prev_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content'],curr_cell.stats_dict['cell_stats']['spatial_information_content']-prev_cell.stats_dict['cell_stats']['spatial_information_content']])
    inp_dict['grid_score'].append([prev_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score'],curr_cell.stats_dict['cell_stats']['grid_score']-prev_cell.stats_dict['cell_stats']['grid_score']])
    inp_dict['b_top'].append([prev_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top'],curr_cell.stats_dict['cell_stats']['b_score_top']-prev_cell.stats_dict['cell_stats']['b_score_top']])
    inp_dict['b_bottom'].append([prev_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom'],curr_cell.stats_dict['cell_stats']['b_score_bottom']-prev_cell.stats_dict['cell_stats']['b_score_bottom']])
    inp_dict['b_right'].append([prev_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right'],curr_cell.stats_dict['cell_stats']['b_score_right']-prev_cell.stats_dict['cell_stats']['b_score_right']])
    inp_dict['b_left'].append([prev_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left'],curr_cell.stats_dict['cell_stats']['b_score_left']-prev_cell.stats_dict['cell_stats']['b_score_left']])
                            
    # inp_dict['speed_score'].append([prev_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score'],curr_cell.stats_dict['cell_stats']['speed_score']-prev_cell.stats_dict['cell_stats']['speed_score']])
    # inp_dict['hd_score'].append([prev_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score'],curr_cell.stats_dict['cell_stats']['hd_score']-prev_cell.stats_dict['cell_stats']['hd_score']])

    return inp_dict

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

def _aggregate_cell_info(study):
    max_centroid_count = 0
    blobs_dict = {}
    type_dict = {}
    spatial_obj_dict = {}
    for animal in study.animals:
        # get largest possible cell id
        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)
        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1
            prev_field_size_len = None 
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                ensemble = ses.get_cell_data()['cell_ensemble']
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)
                    # if 'spatial_spike_train' in cell.stats_dict['cell_stats']:
                    #     spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']
                    # else:
                    #     spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': ses.get_position_data()['position']})

                    spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']

                    image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train)

                    id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)

                    blobs_dict[id] = [image, n_labels, labels, centroids, field_sizes]

                    spatial_obj_dict[id] = spatial_spike_train

                    type_dict[id] = []
                

                    if prev_field_size_len is not None:
                        # print(len(field_sizes))
                        max_centroid_count = max(max_centroid_count, len(field_sizes) * prev_field_size_len)
                    else:
                        prev_field_size_len = len(field_sizes)

    return max_centroid_count, blobs_dict

def _sort_centroids_by_field_size(field_sizes_source, field_sizes_target, blobs_map_source, blobs_map_target, centroids_prev, centroids_curr):
    sort_idx_source = np.argsort(-np.array(field_sizes_source))
    source_labels = np.zeros(blobs_map_source.shape)
    map_dict = {}
    for k in np.unique(blobs_map_source):
        if k != 0:
            # row, col = np.where(labels_prev == k)
            # source_labels[row, col] = sort_idx_source[k-1] + 1
            map_dict[k] = sort_idx_source[k-1] + 1
        else:
            map_dict[k] = 0
    source_labels = np.vectorize(map_dict.get)(blobs_map_source)

    source_centroids = centroids_prev[sort_idx_source]

    # sort_idx_target = np.argsort(-np.array(field_sizes_target))
    # target_centers = target_centers[sort_idx_target]

    sort_idx_target = np.argsort(-np.array(field_sizes_target))
    target_labels = np.zeros(blobs_map_target.shape)
    map_dict = {}
    for k in np.unique(blobs_map_target):
        if k != 0:
            map_dict[k] = sort_idx_target[k-1] + 1
        else:
            map_dict[k] = 0
    target_labels = np.vectorize(map_dict.get)(blobs_map_target)

    target_centroids = centroids_curr[sort_idx_target]

    return target_labels, source_labels, source_centroids, target_centroids

def _fill_centroid_dict(centroid_dict, max_centroid_count, wass_args, centroid_pairs):
    field_wass, centroid_wass, binary_wass = wass_args
    for n in range(max_centroid_count):
        wass_key = 'centroid_wass_'+str(n+1)
        id_key = 'centroid_ids_'+str(n+1)

        if wass_key not in centroid_dict:
            centroid_dict[wass_key] = []
            centroid_dict[id_key] = []

        if n < len(centroid_wass):
            centroid_dict[wass_key].append([field_wass[n], centroid_wass[n], binary_wass[n]])
            centroid_dict[id_key].append(centroid_pairs[n])
        else:
            centroid_dict[wass_key].append([0,0,0])
            centroid_dict[id_key].append([0,0])

    return centroid_dict