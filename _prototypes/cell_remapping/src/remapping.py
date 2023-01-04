import os, sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.settings import rate_output, obj_output
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_rate_remapping
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, compute_wasserstein_distance, single_point_wasserstein, pot_sliced_wasserstein
from _prototypes.cell_remapping.src.masks import make_object_ratemap, check_disk_arena, flat_disk_mask
# from library.maps.map_utils import disk_mask

"""

TODO (in order of priority)

- Pull out POT dependecies for sliced wass - DONE
- take all spikes in cell, get (x,y) position, make sure they are scaled properly (i.e. in cm/inches) at file loading part - DONE
- read ppm from position file ('pixels_per_meter' word search) and NOT settings.py file - DONE
- check ppm can bet set at file loading, you likely gave that option if 'ppm' not present in settings - DONE

- Use map blobs to get fields
- get idx in fields and calculate euclidean distance for all permutations of possible field combinations
- can start with only highest density fields
- MUST revisit map blobs and how the 90th percentile is being done
- Reconcile definition of fields with papers Abid shared in #code to make field definition for our case concrete
- visualize selected fields (plot ratemap + circle/highlight in diff color idx of each field, can plot binary + ratemap below to show true density in field)

- Implement rotation remapping, get ready for case where from session to session field map is rotated by 0/90/180 etc instead of object location
- Will have to do the same as object case where you do every rotation permutation and store the 'true' rotation angle to look at wass distances 

- Implement globabl remapping? Just average ratemaps across all cells in session and use 'average' ratemap of each sessionn in sliced wass

"""


def compute_remapping(study, settings):

    c = 0

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
                            key = 'wass_' + str(var)
                            sliced_key = 'sliced_wass_' + str(var)
                            obj_wass_key = 'obj_wass_' + str(var)

                            object_ratemap, object_pos = make_object_ratemap(var, rate_map_obj)

                            if var == object_location:
                                true_object_pos = object_pos
                                true_object_ratemap = object_ratemap
                            
                            # disk mask fake object ratemap
                            ###### TEMPORARILY TESTING THIS, LIKELY DONT NEED TO DISK MASK THE ARTIFICAL OBJECT RATEMAP
                            if cylinder:
                                object_ratemap = flat_disk_mask(object_ratemap)

                            # print(object_ratemap)
                            num_proj = 100

                            obj_wass = single_point_wasserstein(object_pos, rate_map_obj)

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

                    if prev is not None:
                        # num_proj = 100
                        wass, _, _ = compute_wasserstein_distance(prev, curr)
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
                        rate_output['wass'].append(wass)
                        rate_output['sliced_wass'].append(sliced_wass)

                        plot_rate_remapping(prev, curr, rate_output)

                        # global remapping ?
                        # centroids cdist
                        # nswe 4 direction distance

                        remapping_distances[i-1,cell_label-1] = wass

                        remapping_indices[cell_label-1].append(i-1)

                        remapping_session_ids[cell_label-1].append([i-1,i])

                        c += 1
                    else:
                        prev = np.copy(curr)
                        prev_spatial_spike_train = curr_spatial_spike_train

            if 'rate_remapping' not in animal.stats_dict:
                animal.stats_dict['rate_remapping'] = {}

            distances = np.asarray(remapping_distances[:, cell_label-1])[remapping_indices[cell_label-1]]

            session_pairs = np.asarray(remapping_session_ids[cell_label-1])

            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)] = {}
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['distances'] = distances
            animal.stats_dict['rate_remapping']['cell_' + str(cell_label)]['session_pairs'] = session_pairs

            c += 1

    return {'rate': rate_output, 'object': obj_output}



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