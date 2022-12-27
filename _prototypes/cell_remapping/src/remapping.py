import os, sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.cell_remapping.src.settings import rate_output, obj_output
from library.hafting_spatial_maps import SpatialSpikeTrain2D
from _prototypes.cell_remapping.src.rate_map_plots import plot_obj_remapping, plot_rate_remapping
from _prototypes.cell_remapping.src.wasserstein_distance import sliced_wasserstein, compute_wasserstein_distance
from _prototypes.cell_remapping.src.masks import make_object_ratemap, flat_disk_mask, check_disk_arena


def compute_remapping(study, settings):

    # if study is None:
    #     assert len(paths) > 0 and len(settings) > 0
    #     # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
    #     study = make_study(paths, settings)
    #     # make animals
    #     study.make_animals()
    # elif isinstance(study, Study):
    #     study.make_animals()

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
                cylinder = check_disk_arena(path)

                ### TEMPORARY WAY TO READ OBJ LOC FROM FILE NAME ###
                if settings['hasObject']:
                    object_location = _read_location_from_file(path)

                if j == 0:
                    assert 'matched' in ses.session_metadata.file_paths['cut'], 'Matched cut file was not used for data loading, cannot proceed with non matched cut file as cluster/cell labels are not aligned'

                pos_obj = ses.get_position_data()['position']

                ensemble = ses.get_cell_data()['cell_ensemble']

                # Check if cell id we're iterating through is present in the ensemble of this sessions
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)

                    spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})

                    rate_map_obj = spatial_spike_train.get_map('rate')
                    rate_map, _ = rate_map_obj.get_rate_map(new_size=3)
                    
                    # Disk mask ratemap
                    if cylinder:
                        curr = flat_disk_mask(rate_map)
                    else:
                        curr = rate_map

                    if settings['hasObject']:

                        variations = [0,90,180,270,'no']

                        # compute object remapping for every object position, actual object location is store alongside wass for each object ratemap
                        for var in variations:
                            key = 'wass_' + str(var)
                            sliced_key = 'sliced_wass_' + str(var)

                            object_ratemap, object_pos = make_object_ratemap(var, rate_map_obj)

                            if var == object_location:
                                true_object_pos = object_pos
                            
                            # disk mask fake object ratemap
                            if cylinder:
                                object_ratemap = flat_disk_mask(object_ratemap)

                            print(object_ratemap)
                            num_proj = 100
                            sliced_wass = sliced_wasserstein(object_ratemap, curr, num_proj)
                            wass, _, _ = compute_wasserstein_distance(object_ratemap, curr)

                            obj_output[key].append(wass)
                            obj_output[sliced_key].append(sliced_wass)

                        # Store true obj location
                        obj_output['object_location'].append(object_location)

                        # if object_pos is not None:
                        obj_output['obj_pos_x'].append(true_object_pos['x'])
                        obj_output['obj_pos_y'].append(true_object_pos['y'])
                        # else:
                        #     obj_output['obj_x_pos'].append(None)
                        #     obj_output['obj_y_pos'].append(None)

                        obj_output['animal_id'].append(animal.animal_id)
                        obj_output['unit_id'].append(cell_label)
                        obj_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        obj_output['session_id'].append(seskey)

                        plot_obj_remapping(object_ratemap, curr, obj_output)

                    if prev is not None:
                        num_proj = 100
                        wass, _, _ = compute_wasserstein_distance(prev, curr)
                        sliced_wass = sliced_wasserstein(prev, curr, num_proj)

                        rate_output['animal_id'].append(animal.animal_id)
                        rate_output['unit_id'].append(cell_label)
                        rate_output['tetrode'].append(animal.animal_id.split('tet')[-1])
                        rate_output['session_ids'].append(['session_' + str(i), 'session_' + str(i+1)])
                        rate_output['wass'].append(wass)
                        rate_output['sliced_wass'].append(sliced_wass)

                        plot_rate_remapping(prev, curr, rate_output)


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

    return {'rate': rate_output, 'object': obj_output}



def _read_location_from_file(path):
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

    return object_location