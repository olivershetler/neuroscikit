import os, sys
import numpy as np
import itertools
import re
import matplotlib.pyplot as plt

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_rate_remapping, plot_fields_remapping
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, single_point_wasserstein, pot_sliced_wasserstein, compute_centroid_remapping, _get_ratemap_bucket_midpoints
from _prototypes.cell_remapping.src.masks import make_object_ratemap, check_disk_arena, flat_disk_mask
from library.maps import map_blobs
from scripts.batch_map.batch_map import batch_map 
from _prototypes.cell_remapping.src.settings import obj_output, centroid_output, tasks, session_comp_categories, regular_output, context_output, variations
from scripts.batch_map.LEC_naming import LEC_naming_format, extract_name

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

                                
def _check_single_format(filename, format, fxn):
    print(str(format), str(filename))
    if re.match(str(format), str(filename)) is not None:
        return fxn(filename)
    

def compute_remapping(study, settings, data_dir):

    c = 0

    batch_map(study, tasks, ratemap_size=settings['ratemap_dims'][0])
    
    max_centroid_count, blobs_dict = _aggregate_cell_info(study, settings)

    centroid_dict = centroid_output
    regular_dict = regular_output
    context_dict = context_output
    obj_dict = obj_output

    for animal in study.animals:

        # if settings['useMatchedCut']:
        #     # get largest possible cell id
        #     max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)
        # else:
        #     max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))

        max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))

        # print('max matched cell count: ' + str(max_matched_cell_count))
        # for x in animal.sessions:
        #     print(x)
        #     print('ensemble label ids: ' + str(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()))

        # len(session) - 1 bcs thats number of comparisons. e.g. 3 session: ses1-ses2, ses2-ses3 so 2 distances will be given for remapping
        remapping_distances = np.zeros((len(list(animal.sessions.keys()))-1, max_matched_cell_count))
        remapping_indices = [[] for k in range(max_matched_cell_count)]
        remapping_session_ids = [[] for k in range(max_matched_cell_count)]

        # for every existing cell id across all sessions
        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1
            print('Cell ' + str(cell_label))

            # prev ratemap
            prev = None
            # comp_prev = None

            # for every session
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                print(seskey)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet']
                fname = path.split('/')[-1].split('.')[0]

                # Check if cylinder
                cylinder, true_var = check_disk_arena(fname)

                group, name = extract_name(fname)

                formats = LEC_naming_format[group][name][settings['type']]

                for format in list(formats.keys()):
                    checked = _check_single_format(fname, format, formats[format])
                    if checked is not None:
                        break
                    else:
                        continue

                stim, depth, name, date = checked

                ### TEMPORARY WAY TO READ OBJ LOC FROM FILE NAME ###
                if settings['hasObject']:
                    # object_location = _read_location_from_file(path, cylinder, true_var)

                    object_location = stim
                    
                    if object_location != 'NO':
                        object_location = int(object_location)

                ensemble = ses.get_cell_data()['cell_ensemble']

                # Check if cell id we're iterating through is present in the ensemble of this sessions
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)

                    # spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})

                    spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    if settings['normalizeRate']:
                        rate_map, _ = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])
                    else:
                        _, rate_map = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])

                    assert rate_map.shape == (settings['ratemap_dims'][0], settings['ratemap_dims'][1]), 'Wrong ratemap shape {} vs settings shape {}'.format(rate_map.shape, (settings['ratemap_dims'][0], settings['ratemap_dims'][1]))
                    
                    # Disk mask ratemap
                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                        # curr_plot = disk_mask(rate_map)
                    else:
                        curr = rate_map
                        # curr_plot = curr
                    
                    curr_cell = cell
                    curr_spatial_spike_train = spatial_spike_train
                    curr_key = seskey
                    curr_path = ses.session_metadata.file_paths['tet'].split('/')[-1].split('.')[0]
                    curr_id = str(animal.animal_id) + '_' + str(seskey) + '_' + str(cell.cluster.cluster_label)

                    y, x = curr.shape
                    h, w = rate_map_obj.arena_size
                    bin_area = h/y * w/x

                    # If object used in experiment 
                    if settings['hasObject']:

                        _, _, labels, centroids, field_sizes = blobs_dict[curr_id]
                                    
                        labels_curr = np.copy(labels)
                        labels_curr[np.isnan(curr)] = 0
                        val_r, val_c = np.where(labels_curr == 1)
                        c_count = len(np.unique(labels_curr)) - 1

                        # take only field label = 1 = largest field = main
                        main_field_coverage = len(np.where(labels_curr == 1)[0])/len(np.where(~np.isnan(curr))[0])
                        main_field_area = len(np.where(labels_curr == 1)[0])
                        main_field_rate = np.sum(curr[val_r, val_c])

                        # make all field labels = 1
                        labels_curr[labels_curr != 0] = 1
                        val_r, val_c = np.where(labels_curr == 1)
                        # cumulative_coverage = np.max(field_sizes)
                        cumulative_coverage = len(np.where(labels_curr != 0)[0])/len(np.where(~np.isnan(curr))[0])
                        cumulative_area = len(np.where(labels_curr == 1)[0])
                        cumulative_rate = np.sum(curr[val_r, val_c])

                        height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(rate_map_obj.arena_size, y, x)
                        
                        # ['whole', 'field', 'bin', 'centroid']
                        for obj_score in settings['object_scores']:

                            # # Possible object locations (change to get from settings)
                            # variations = [0,90,180,270,'no']
                            # compute object remapping for every object position, actual object location is store alongside wass for each object ratemap
                            for var in variations:
                                obj_wass_key = 'obj_wass_' + str(var)

                                object_ratemap, object_pos = make_object_ratemap(var, rate_map_obj, new_size=settings['ratemap_dims'][0])

                                if cylinder:
                                    object_ratemap = flat_disk_mask(object_ratemap)
                                    # ids where not nan
                                    row, col = np.where(~np.isnan(object_ratemap))
                                    disk_ids = np.array([row, col]).T
                                else:
                                    disk_ids = None

                                if var == object_location:
                                    true_object_pos = object_pos
                                    true_object_ratemap = object_ratemap

                                if isinstance(object_pos, dict):
                                    obj_x = width_bucket_midpoints[object_pos['x']]
                                    obj_y = height_bucket_midpoints[object_pos['y']]
                                else:
                                    obj_y = height_bucket_midpoints[object_pos[0]]
                                    obj_x = width_bucket_midpoints[object_pos[1]] 

                                if var == 'no':
                                    y, x = object_ratemap.shape
                                    # find indices of not nan 
                                    row_prev, col_prev = np.where(~np.isnan(object_ratemap))
                                    row_curr, col_curr = np.where(~np.isnan(curr))

                                    assert row_prev.all() == row_curr.all() and col_prev.all() == col_curr.all(), 'Nans in different places'

                                    height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(curr_spatial_spike_train.arena_size, y, x)
                                    height_bucket_midpoints = height_bucket_midpoints[row_curr]
                                    width_bucket_midpoints = width_bucket_midpoints[col_curr]
                                    source_weights = np.array(list(map(lambda x, y: object_ratemap[x,y], row_curr, col_curr)))
                                    target_weights = np.array(list(map(lambda x, y: curr[x,y], row_curr, col_curr)))
                                    source_weights = source_weights / np.sum(source_weights)
                                    target_weights = target_weights / np.sum(target_weights)
                                    coord_buckets = np.array(list(map(lambda x, y: [height_bucket_midpoints[x],width_bucket_midpoints[y]], row_curr, col_curr)))
                
                                # EMD on norm/unnorm ratemap + object map for OBJECT remapping
                                if obj_score == 'whole':
                                    if var == 'no':
                                        obj_wass = pot_sliced_wasserstein(coord_buckets, coord_buckets, source_weights, target_weights, n_projections=settings['n_projections'])
                                    else:
                                        obj_wass = single_point_wasserstein(object_pos, curr, rate_map_obj.arena_size, ids=disk_ids)

                                elif obj_score == 'field':

                                    # field ids is ids of binary field/map blolb

                                    # TAKE ONLY MAIN FIELD --> already sorted by size
                                    row, col = np.where(labels_curr == 1)
                                    field_ids = np.array([row, col]).T

                                    # take ids that are both in disk and in field
                                    field_disk_ids = np.array([x for x in field_ids if x in disk_ids])
                                    
                                    if var == 'no':
                                        obj_wass = pot_sliced_wasserstein(coord_buckets, coord_buckets[field_disk_ids], source_weights, target_weights[field_disk_ids], n_projections=settings['n_projections'])
                                    else:
                                        # NEED TO NORMALIZE FIELD NOT WHOLE MAP
                                        # save total mass in firing field, save area for each field
                                        # remember long format
                                        obj_wass = single_point_wasserstein(object_pos, curr, rate_map_obj.arena_size, ids=field_disk_ids)

                                elif obj_score == 'binary':
                                    # obj_bin_wass = single_point_wasserstein(object_pos, labels_curr, rate_map_obj.arena_size, ids=field_ids)

                                    if var == 'no':
                                        row, col = np.where(~np.isnan(object_ratemap))
                                        height_source_pts = height_bucket_midpoints[row]
                                        width_source_pts = width_bucket_midpoints[col]
                                        source_pts = np.array([height_source_pts, width_source_pts]).T
                                    else:
                                        source_pts = np.array([obj_y, obj_x]).reshape(1,2)

                                    curr_spike_pos_x, curr_spike_pos_y, _ = curr_spatial_spike_train.get_spike_positions()
                                    curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).T

                                    obj_wass = pot_sliced_wasserstein(source_pts, curr_pts, n_projections=settings['n_projections'])

                                elif obj_score == 'centroid':
                                    # if c_count > 1:
                                    #     cumulative_centroid = np.mean(centroids, axis=0)
                                    # else:

                                    # TAKE ONLY MAIN FIELD --> already sorted by size
                                    main_centroid = centroids[0]

                                    # print(cumulative_centroid, centroids, c_count)

                                    # euclidean distance between point
                                    obj_wass = np.linalg.norm(np.array((obj_y, obj_x)) - np.array((main_centroid[0],main_centroid[1])))

                                # obj_dict[obj_wass_key].append([obj_wass, obj_field_wass, obj_bin_wass, c_wass])
                                obj_dict[obj_wass_key].append(obj_wass)
                            
                            obj_dict['score'].append(obj_score)
                            obj_dict['main_field_coverage'].append(main_field_coverage)
                            obj_dict['main_field_area'].append(main_field_area)
                            obj_dict['main_field_rate'].append(main_field_rate)
                            obj_dict['cumulative_coverage'].append(cumulative_coverage)
                            obj_dict['cumulative_area'].append(cumulative_area)
                            obj_dict['cumulative_rate'].append(cumulative_rate)
                            obj_dict['field_count'].append(c_count)
                            obj_dict['bin_area'].append(bin_area[0])

                            # Store true obj location
                            obj_dict['object_location'].append(object_location)

                            # if object_pos is not None:
                            obj_dict['obj_pos_x'].append(true_object_pos['x'])
                            obj_dict['obj_pos_y'].append(true_object_pos['y'])

                            obj_dict['signature'].append(curr_path)
                            obj_dict['name'].append(name)
                            obj_dict['date'].append(date)
                            obj_dict['depth'].append(depth)
                            obj_dict['unit_id'].append(cell_label)
                            obj_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                            obj_dict['session_id'].append(seskey)

                        if settings['plotObject']:
                            plot_obj_remapping(true_object_ratemap, curr, obj_dict, data_dir)

                    # If prev ratemap is not None (= we are at session2 or later, session1 has no prev session to compare)
                    if prev is not None:

                        # get x and y pts for spikes in pair of sessions (prev and curr) for a given comparison

                        prev_spike_pos_x, prev_spike_pos_y, _ = prev_spatial_spike_train.get_spike_positions()
                        prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).T

                        curr_spike_pos_x, curr_spike_pos_y, _ = curr_spatial_spike_train.get_spike_positions()
                        curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).T

                        y, x = prev.shape
                        # find indices of not nan 
                        row_prev, col_prev = np.where(~np.isnan(prev))
                        row_curr, col_curr = np.where(~np.isnan(curr))

                        assert row_prev.all() == row_curr.all() and col_prev.all() == col_curr.all(), 'Nans in different places'

                        height_bucket_midpoints, width_bucket_midpoints = _get_ratemap_bucket_midpoints(prev_spatial_spike_train.arena_size, y, x)
                        height_bucket_midpoints = height_bucket_midpoints[row_curr]
                        width_bucket_midpoints = width_bucket_midpoints[col_curr]
                        source_weights = np.array(list(map(lambda x, y: prev[x,y], row_curr, col_curr)))
                        target_weights = np.array(list(map(lambda x, y: curr[x,y], row_curr, col_curr)))
                        source_weights = source_weights / np.sum(source_weights)
                        target_weights = target_weights / np.sum(target_weights)
                        coord_buckets = np.array(list(map(lambda x, y: [height_bucket_midpoints[x],width_bucket_midpoints[y]], row_curr, col_curr)))

                        # source_coords = np.array(list(map(lambda x: prev[x[0],x[1]], coord_buckets)))
                        # target_coords = np.array(list(map(lambda x: curr[x[0],x[1]], coord_buckets)))

                        # # Use cts to 'weigh' number of (x,y) pts contributed by each spike
                        # prev_coords = list(map(lambda i,j: [[height_bucket_midpoints[i], width_bucket_midpoints[j]] for x in range(int(prev_pts_ct[i,j]))], np.arange(0,prev_pts_ct.shape[0],1), np.arange(0,prev_pts_ct.shape[1],1)))
                        # curr_coords = list(map(lambda i,j: [[height_bucket_midpoints[i], width_bucket_midpoints[j]] for x in range(int(curr_pts_ct[i,j]))], np.arange(0,curr_pts_ct.shape[0],1), np.arange(0,curr_pts_ct.shape[1],1)))

                        # prev_coords = list(itertools.chain.from_iterable(prev_coords))
                        # curr_coords = list(itertools.chain.from_iterable(curr_coords))

                        # print(coord_buckets.shape, source_weights.shape, target_weights.shape)

                        for rate_score in settings['rate_scores']:

                            if rate_score == 'binary':
                                # This is EMD on whole map (binary) 
                                wass = pot_sliced_wasserstein(prev_pts, curr_pts, n_projections=settings['n_projections'])
                            elif rate_score == 'whole':
                                # This is EMD on whole map for normalized/unnormalized rate remapping
                                wass = pot_sliced_wasserstein(coord_buckets, coord_buckets, source_weights, target_weights, n_projections=settings['n_projections'])
                                                    
                            regular_dict['signature'].append([prev_path, curr_path])
                            regular_dict['name'].append(name)
                            regular_dict['date'].append(date)
                            regular_dict['depth'].append(depth)
                            regular_dict['unit_id'].append(cell_label)
                            regular_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                            regular_dict['session_ids'].append([prev_key, curr_key])
                            # regular_dict['session_paths'].append([prev_path, curr_path])
                            # regular_dict['sliced_wass'].append(sliced_wass)
                            # regular_dict['bin_wass'].append(wass)
                            regular_dict['score'].append(rate_score)
                            regular_dict['wass'].append(wass)


                        if settings['plotRegular']:
                            plot_rate_remapping(prev, curr, regular_dict, data_dir)

                        if settings['runFields']:
                            # blobs_dict_curr = curr_cell.stats['field_size_data']
                            # blobs_dict_prev = prev_cell.stats['field_size_data']

                            image_prev, n_labels_prev, source_labels, source_centroids, field_sizes_prev = blobs_dict[prev_id]

                            image_curr, n_labels_curr, target_labels, target_centroids, field_sizes_curr = blobs_dict[curr_id]

                            # target_labels, source_labels, source_centroids, target_centroids = _sort_filter_centroids_by_field_size(prev, curr, field_sizes_prev, field_sizes_curr, labels_prev, labels_curr, centroids_prev, centroids_curr, prev_spatial_spike_train.arena_size)

                            # assert np.unique(target_labels).all() == np.unique(labels_curr).all()
                            # assert np.unique(source_labels).all() == np.unique(labels_prev).all()

                            # prev spatial spike train is source spatial spike train
                            # field_wass, field_pairs, cumulative_wass, test_wass, centroid_wass, binary_wass = compute_centroid_remapping(target_labels, source_labels, curr_spatial_spike_train, prev_spatial_spike_train, target_centroids, source_centroids)
                            # cumulative_wass = compute_cumulative_centroid_remapping(target_centers, source_labels, prev_spatial_spike_train, field_sizes_curr)
                            # print(animal.animal_id, cell_label, animal.animal_id.split('tet')[-1], [prev_key, curr_key])
                            
                            if len(np.unique(target_labels)) > 1 and len(np.unique(source_labels)) > 1:

                                """
                                cumulative_dict has field/centroid/binary wass, cumulative = all fields used
                                'field_wass' is EMD on ALL fields for norm/unnorm RATE remapping
                                'centroid_wass' is EMD on ALL field centroids for norm/unnorm LOCATION remapping (i.e. field centre points averaged + EMD calculated)
                                'binary_wass' is EMD on ALL fields (binary) for norm/unnorm LOCATION remapping (i.e. unweighted such that each pt contributes equally within field)

                                permute_dict has field/centroid/binary wass, permute = all combinations of single fields for given session pair
                                'field_wass' is EMD on SINGLE fields for norm/unnorm RATE remapping
                                'centroid_wass' is EMD on SINGLE field centroids for norm/unnorm LOCATION remapping (i.e. EMD calculated directly between diff centroid pairs across sessions)
                                'binary_wass' is EMD on SINGLE fields (binary) for norm/unnorm LOCATION remapping (i.e. unweighted such that each pt contributes equally within field)
                                """
                                permute_dict, cumulative_dict = compute_centroid_remapping(target_labels, source_labels, curr_spatial_spike_train, prev_spatial_spike_train, target_centroids, source_centroids, settings)

                                y, x = curr.shape
                                h, w = rate_map_obj.arena_size
                                bin_area = h/y * w/x
                                field_count = [len(np.unique(source_labels)) - 1,len(np.unique(target_labels)) - 1]

                                for centroid_score in settings['centroid_scores']:

                                    score_key = centroid_score + '_wass'

                                    centroid_dict['score'].append(centroid_score)
                                    # centroid_dict['signature'].append(curr_path)
                                    centroid_dict['name'].append(name)
                                    centroid_dict['date'].append(date)
                                    centroid_dict['depth'].append(depth)
                                    centroid_dict['bin_area'].append(bin_area[0])
                                    centroid_dict['field_count'].append(field_count)
                                    centroid_dict['unit_id'].append(cell_label)
                                    centroid_dict['tetrode'].append(animal.animal_id.split('tet')[-1])
                                    centroid_dict['session_ids'].append([prev_key, curr_key])
                                    centroid_dict['signature'].append([prev_path, curr_path])

                                    # field wass is weighed, centroid wass is centre pts, binary wass is unweighed (as if was bianry map with even weight so one pt per position in map)
                                    # centroid_dict['cumulative_wass'].append([cumulative_dict['field_wass'], cumulative_dict['centroid_wass'], cumulative_dict['binary_wass']])
                                    centroid_dict['cumulative_wass'].append(cumulative_dict[score_key])

                                    copy_labels = np.copy(source_labels)
                                    copy_labels[np.isnan(prev)] = 0
                                    copy_labels[copy_labels != 0] = 1
                                    val_r, val_c = np.where(copy_labels == 1)
                                    # cumulative_source_coverage = np.max(field_sizes_prev)
                                    cumulative_source_coverage = len(np.where(copy_labels != 0)[0])/len(np.where(~np.isnan(prev))[0])
                                    cumulative_source_area = len(np.where(copy_labels == 1)[0])
                                    cumulative_source_rate = np.sum(prev[val_r, val_c])

                                    copy_labels = np.copy(target_labels)
                                    copy_labels[np.isnan(curr)] = 0
                                    copy_labels[copy_labels != 0] = 1
                                    val_r, val_c = np.where(copy_labels == 1)
                                    # cumulative_target_coverage = np.max(field_sizes_curr)
                                    cumulative_target_coverage = len(np.where(copy_labels != 0)[0])/len(np.where(~np.isnan(curr))[0])
                                    cumulative_target_area = len(np.where(copy_labels == 1)[0])
                                    cumulative_target_rate = np.sum(curr[val_r, val_c])

                                    # cumulative_source_coverage = np.sum(field_sizes_prev)
                                    # cumulative_source_area = len(np.where(source_labels != 0)[0])
                                    # cumulative_source_rate = np.sum(prev[source_labels != 0])
                                    # cumulative_target_coverage = np.sum(field_sizes_curr)
                                    # cumulative_target_area = len(np.where(target_labels != 0)[0])
                                    # cumulative_target_rate = np.sum(curr[target_labels != 0])

                                    centroid_dict['cumulative_coverage'].append([cumulative_source_coverage, cumulative_target_coverage])
                                    centroid_dict['cumulative_area'].append([cumulative_source_area, cumulative_target_area])
                                    centroid_dict['cumulative_rate'].append([cumulative_source_rate, cumulative_target_rate])

                                                            # wass_args = [permute_dict['field_wass'], permute_dict['binary_wass'], permute_dict['centroid_wass']]
                                    wass_to_add = permute_dict[score_key]

                                    # need to make test_wass save as in fill centroid dict
                                    # centroid_dict['test_wass'].append(test_wass)
                                    # centroid_dict['centroid_wass'].append(centroid_wass)

                                    centroid_dict = _fill_centroid_dict(centroid_dict, max_centroid_count, wass_to_add, permute_dict['pairs'], prev, source_labels, field_sizes_prev, curr, target_labels, field_sizes_curr)

                                if settings['plotFields']:
                                    if cylinder:
                                        target_labels = flat_disk_mask(target_labels)
                                        source_labels = flat_disk_mask(source_labels)
                                    plot_fields_remapping(source_labels, target_labels, prev_spatial_spike_train, curr_spatial_spike_train, source_centroids, target_centroids, centroid_dict, data_dir, settings, cylinder=cylinder)

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                        c += 1

                    prev = curr
                    prev_spatial_spike_train = curr_spatial_spike_train
                    prev_id = curr_id
                    prev_cell = curr_cell
                    prev_key = curr_key
                    prev_path = curr_path
                    # prev_plot = curr_plot
            
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
                        path = ses.session_metadata.file_paths['tet']
                        fname = path.split('/')[-1].split('.')[0]
                        cylinder, true_var = check_disk_arena(fname)

                        ensemble = ses.get_cell_data()['cell_ensemble']

                        if cell_label in ensemble.get_cell_label_dict():
                            cell = ensemble.get_cell_by_id(cell_label)

                            spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']

                            comp_curr = spatial_spike_train
                            comp_curr_cell = cell
                            curr_key = seskey
                            curr_path = ses.session_metadata.file_paths['tet'].split('/')[-1].split('.')[0]

                            if comp_prev is not None:
                                prev_spike_pos_x, prev_spike_pos_y, _ = comp_prev.get_spike_positions()
                                # prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).reshape((-1,2))
                                prev_pts = np.array([prev_spike_pos_x, prev_spike_pos_y]).T

                                curr_spike_pos_x, curr_spike_pos_y, _ = comp_curr.get_spike_positions()
                                # curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).reshape((-1,2))
                                curr_pts = np.array([curr_spike_pos_x, curr_spike_pos_y]).T

                                sliced_wass = pot_sliced_wasserstein(prev_pts, curr_pts, n_projections=settings['n_projections'])

                                context_dict[categ]['signature'].append(curr_path)
                                context_dict[categ]['name'].append(name)
                                context_dict[categ]['date'].append(date)
                                context_dict[categ]['depth'].append(depth)
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
                            prev_path = curr_path
                            

            c += 1

    return {'regular': regular_dict, 'object': obj_dict, 'centroid': centroid_dict, 'context': context_dict}

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
        object_location = items[idx].split('.')[0].split('_')[0].lower()

    object_present = True
    if str(object_location) == 'no':
        object_present == False
        object_location = 'no'
    elif str(object_location) == 'zero':
        object_location = 0
    else:
        object_location = int(object_location)
        assert int(object_location) in [0,90,180,270], 'Failed bcs obj location is ' + str(int(object_location)) + ' and that is not in [0,90,180,270]'

    return object_location

def _aggregate_cell_info(study, settings):
    ratemap_size = settings['ratemap_dims'][0]
    max_centroid_count = 0
    blobs_dict = {}
    type_dict = {}
    spatial_obj_dict = {}
    for animal in study.animals:
        # get largest possible cell id
        # max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)
        max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))
        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1
            prev_field_size_len = None 
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                ensemble = ses.get_cell_data()['cell_ensemble']

                path = ses.session_metadata.file_paths['tet']
                fname = path.split('/')[-1].split('.')[0]

                # Check if cylinder
                cylinder, true_var = check_disk_arena(fname)
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)
                    # if 'spatial_spike_train' in cell.stats_dict['cell_stats']:
                    #     spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']
                    # else:
                    #     spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': ses.get_position_data()['position']})

                    spatial_spike_train = cell.stats_dict['cell_stats']['spatial_spike_train']

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    if settings['normalizeRate']:
                        rate_map, _ = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])
                    else:
                        _, rate_map = rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])

                    assert rate_map.shape == (settings['ratemap_dims'][0], settings['ratemap_dims'][1]), 'Wrong ratemap shape {} vs settings shape {}'.format(rate_map.shape, (settings['ratemap_dims'][0], settings['ratemap_dims'][1]))
                    
                    if cylinder:
                        rate_map = flat_disk_mask(rate_map)

                    image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train, ratemap_size=ratemap_size, cylinder=cylinder)

                    labels, centroids, field_sizes = _sort_filter_centroids_by_field_size(rate_map, field_sizes, labels, centroids, spatial_spike_train.arena_size)

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


# def _sort_filter_centroids_by_field_size(prev, curr, field_sizes_source, field_sizes_target, blobs_map_source, blobs_map_target, centroids_prev, centroids_curr, arena_size):
def _sort_filter_centroids_by_field_size(rate_map, field_sizes, blobs_map, centroids, arena_size):
    height, width = arena_size
    y, x = rate_map.shape
    heightStep = height/y
    widthStep = width/x

    bin_area = heightStep * widthStep

    if type(bin_area) == list:
        assert len(bin_area) == 1
        bin_area = bin_area[0]

    # print('HEREEE')
    # print(np.unique(blobs_map_source), np.unique(blobs_map_target), field_sizes_source, field_sizes_target)
    # print(np.argsort(-np.array(field_sizes_source)))

    # sort_idx_source = np.argsort(-np.array(field_sizes_source))

    # find not nan in prev/curr 
    row, col = np.where(np.isnan(rate_map))
    blobs_map[row, col] = 0

    # find bins for each label
    pks = []
    avgs = []
    lbls = []
    for k in np.unique(blobs_map):
        if k != 0:
            row, col = np.where(blobs_map == k)
            
            field = rate_map[row, col]

            # pk_rate = np.max(field)

            # get rate at centroid 
            c_row, c_col = centroids[k-1]
            # convert from decimals to bin by rounding to nearest int
            c_row = int(np.round(c_row))
            c_col = int(np.round(c_col))

            pk_rate = rate_map[c_row, c_col]
            avg_rate = np.mean(field)

            pks.append(pk_rate)
            avgs.append(avg_rate)
            lbls.append(k-1)
    
    pks = np.array(pks)
    ids = np.argsort(-pks)
    sort_idx = np.array(lbls)[ids]

    source_labels = np.zeros(blobs_map.shape)
    map_dict = {}
    for k in np.unique(blobs_map):
        if k != 0:
            row, col = np.where(blobs_map == k)
            # print(k,len(row), len(col), bin_area)
            if (len(row) + len(col)) * bin_area > 22.5:
                idx_to_move_to = np.where(sort_idx == k-1)[0][0]
                # map_dict[k] = sort_idx_source[k-1] + 1
                map_dict[k] = idx_to_move_to + 1
            else:
                print('Blob filtered out with size less than 22.5 cm^2')
                map_dict[k] = 0
        else:
            map_dict[k] = 0
    source_labels = np.vectorize(map_dict.get)(blobs_map)

    source_centroids = np.asarray(centroids)[sort_idx]
    source_field_sizes = np.asarray(field_sizes)[sort_idx]

    return source_labels, source_centroids, source_field_sizes

def _fill_centroid_dict(centroid_dict, max_centroid_count, wass_to_add, centroid_pairs, source_map, source_labels, source_field_sizes, target_map, target_labels, target_field_sizes):
    # field_wass, centroid_wass, binary_wass = wass_args
    # to_add = wass_args
    for n in range(max_centroid_count):
        wass_key = 'c_wass_'+str(n+1)
        id_key = 'c_ids_'+str(n+1)
        coverage_key = 'c_coverage_'+str(n+1)
        area_key = 'c_area_'+str(n+1)
        rate_key = 'c_rate_'+str(n+1)

        if wass_key not in centroid_dict:
            centroid_dict[wass_key] = []
            centroid_dict[id_key] = []
            centroid_dict[coverage_key] = []
            centroid_dict[area_key] = []
            centroid_dict[rate_key] = []

        if n < len(wass_to_add):
            centroid_dict[wass_key].append(wass_to_add[n])
            centroid_dict[id_key].append(centroid_pairs[n])

            labels_curr = np.copy(source_labels)
            labels_curr[np.isnan(source_map)] = 0
            val_r, val_c = np.where(labels_curr == centroid_pairs[n][0])
            # take only field label = 1 = largest field = main
            # source_field_coverage = source_field_sizes[centroid_pairs[n][0]-1]
            source_field_coverage = len(np.where(labels_curr == centroid_pairs[n][0])[0])/len(np.where(~np.isnan(source_labels))[0])
            source_field_area = len(np.where(labels_curr == centroid_pairs[n][0])[0])
            source_field_rate = np.sum(source_map[val_r, val_c])

            labels_curr = np.copy(target_labels)
            labels_curr[np.isnan(target_map)] = 0
            val_r, val_c = np.where(labels_curr == centroid_pairs[n][1])
            # take only field label = 1 = largest field = main
            # target_field_coverage = target_field_sizes[centroid_pairs[n][1]-1]
            target_field_coverage = len(np.where(labels_curr == centroid_pairs[n][1])[0])/len(np.where(~np.isnan(target_labels))[0])
            target_field_area = len(np.where(labels_curr == centroid_pairs[n][1])[0])
            target_field_rate = np.sum(target_map[val_r, val_c])


            # row, col = np.where(~np.isnan(source_map))
            # disk_ids = np.array([row, col]).T
            # source_field_coverage = np.max(source_field_sizes)
            # source_field_area = len(np.where(source_labels == centroid_pairs[n][0])[0])
            # row, col = np.where(source_labels == 1)
            # whole_map_ids = np.array([row, col]).T
            # map_disk_ids = set(map(tuple, filter(lambda x: np.all(np.isin(tuple(x), whole_map_ids)), disk_ids))) 
            # source_field_rate = np.sum(source_map[np.array(list(map_disk_ids))[:,0], np.array(list(map_disk_ids))[:,1]])

            # row, col = np.where(~np.isnan(target_map))
            # disk_ids = np.array([row, col]).T
            # target_field_coverage = np.max(target_field_sizes)
            # target_field_area = len(np.where(target_labels == centroid_pairs[n][1])[0])
            # row, col = np.where(target_labels == 1)
            # whole_map_ids = np.array([row, col]).T
            # map_disk_ids = set(map(tuple, filter(lambda x: np.all(np.isin(tuple(x), whole_map_ids)), disk_ids))) 
            # # target_field_rate = np.sum(target_map[row, col])
            # target_field_rate = np.sum(target_map[np.array(list(map_disk_ids))[:,0], np.array(list(map_disk_ids))[:,1]])

            
            # print(len(disk_ids), len(whole_map_ids), len(np.array(list(map_disk_ids))))
            centroid_dict[coverage_key].append([source_field_coverage, target_field_coverage])
            centroid_dict[area_key].append([source_field_area, target_field_area])
            centroid_dict[rate_key].append([source_field_rate, target_field_rate])
            
        else:
            centroid_dict[wass_key].append(np.nan)
            centroid_dict[id_key].append([np.nan])
            centroid_dict[coverage_key].append([np.nan, np.nan])
            centroid_dict[area_key].append([np.nan, np.nan])
            centroid_dict[rate_key].append([np.nan, np.nan])

        # if wass_key not in centroid_dict:
        #     centroid_dict[wass_key] = []
        #     centroid_dict[id_key] = []

        # if n < len(centroid_wass):
        #     centroid_dict[wass_key].append([field_wass[n], centroid_wass[n], binary_wass[n]])
        #     centroid_dict[id_key].append(centroid_pairs[n])
        # else:
        #     centroid_dict[wass_key].append([0,0,0])
        #     centroid_dict[id_key].append([0,0])

    return centroid_dict