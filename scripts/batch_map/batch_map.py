import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from library.study_space import Study, Animal
from library.workspace import Workspace

from library.maps import autocorrelation, binary_map, spatial_tuning_curve, map_blobs
from library.hafting_spatial_maps import HaftingOccupancyMap, HaftingRateMap, HaftingSpikeMap, SpatialSpikeTrain2D
from library.scores import hd_score, grid_score, border_score
from library.scores import rate_map_stats, rate_map_coherence, speed_score
from PIL import Image
import numpy as np
from matplotlib import cm

# class BatchMaps(Workspace):
#     def __init__(self, study: StudyWorkspace):
#         pass


def batch_map(study: Study, tasks: dict):
    """
    Computes rate maps across all animals, sessions, cells in a study.

    Use tasks dictionary as true/false flag with variable to compute
    e.g. {'rate_map': True, 'binary_map': False}
    """

    study.make_animals()
    animals = study.animals

    for animal in animals:

        # cells, waveforms = sort_cell_spike_times(animal)

        # sort_cell_spike_times(animal)

        # cluster = session.get_spike_data['spike_cluster']

        # sort_spikes_by_cell(cluster)

        k = 1

        for session_key in animal.sessions:
            session = animal.sessions[session_key]

            c = 1

            for cell in session.get_cell_data()['cell_ensemble'].cells:

                print('session ' + str(k) + ', cell ' + str(c))
                
                pos_obj = session.get_position_data()['position']

                spatial_spike_train = session.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})
                # stop()

                occ_obj = HaftingOccupancyMap(spatial_spike_train)
                occ_map, _, _ = occ_obj.get_occupancy_map()

                
                spike_obj = HaftingSpikeMap(spatial_spike_train)
                spike_map = spike_obj.get_spike_map()

                rate_obj = HaftingRateMap(spatial_spike_train)
                rate_map, raw_rate_map = rate_obj.get_rate_map()

                ratemap_stats_dict  = rate_map_stats(spatial_spike_train)

                autocorr_map = autocorrelation(spatial_spike_train)

                cell_stats = {}
                cell_stats['rate_map_smooth'] = rate_map
                cell_stats['occupancy_map'] = occ_map
                cell_stats['rate_map_raw'] = raw_rate_map
                cell_stats['autocorrelation_map'] = autocorr_map


                if tasks['binary_map']:
                    binmap = binary_map(spatial_spike_train)
                    binmap_im = Image.fromarray(np.uint8(binmap*255))
                    cell_stats['binary_map'] = binmap
                    cell_stats['binary_map_im'] = binmap_im

                if tasks['autocorrelation_map']:
                    cell_stats['autocorr_map'] = autocorr_map
                    autocorr_map_im = Image.fromarray(np.uint8(cm.jet(autocorr_map)*255))
                    cell_stats['autocorr_map_im'] = autocorr_map_im

                if tasks['sparsity']:
                    cell_stats['ratemap_stats_dict'] = ratemap_stats_dict['sparsity']

                if tasks['selectivity']:
                    cell_stats['ratemap_stats_dict'] = ratemap_stats_dict['selectivity']

                if tasks['information']:
                    cell_stats['ratemap_stats_dict'] = ratemap_stats_dict['spatial_information_content']

                if tasks['coherence']:
                    coherence = rate_map_coherence(spatial_spike_train)
                    cell_stats['coherence'] = coherence

                if tasks['speed_score']:
                    s_score = speed_score(spatial_spike_train)
                    cell_stats['speed_score'] = s_score

                if tasks['hd_score'] or tasks['tuning_curve']:
                    tuned_data, spike_angles, angular_occupancy, bin_array = spatial_tuning_curve(spatial_spike_train)
                    cell_stats['tuned_data'] = tuned_data
                    cell_stats['tuned_data_angles'] = spike_angles
                    cell_stats['angular_occupancy'] = angular_occupancy
                    cell_stats['angular_occupancy_bins'] = bin_array

                if tasks['hd_score']:
                    hd_hist = hd_score(spatial_spike_train)
                    cell_stats['hd_hist'] = hd_hist

                if tasks['grid_score']:
                    true_grid_score = grid_score(spatial_spike_train)
                    cell_stats['grid_score'] = true_grid_score

                if tasks['border_score']:
                    b_score = border_score(spatial_spike_train)
                    cell_stats['b_score_top'] = b_score[0]
                    cell_stats['b_score_bottom'] = b_score[1]
                    cell_stats['b_score_left'] = b_score[2]
                    cell_stats['b_score_right'] = b_score[3]

                if tasks['field_sizes']:
                    image, n_labels, labels, centroids, field_sizes = map_blobs(spatial_spike_train)
                    cell_stats['field_size_data'] = {'image': image, 'n_labels': n_labels, 'labels': labels, 'centroids': centroids, 'field_sizes': field_sizes}

                cell.stats_dict['cell_stats'] = cell_stats

                c += 1
            k += 1