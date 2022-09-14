import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.data_study import Study, Animal

from library.maps import get_rate_map, get_autocorrelation, get_occupancy_map, get_binary_map, compute_tuning_curve, spikePos, get_map_blobs
from library.scores import get_hd_histogram, compute_grid_score, compute_border_score
from library.sort_cell_spike_times import sort_cell_spike_times
from library.spatial import speed2D

from opexebo.analysis import rate_map_stats, rate_map_coherence, speed_score
from PIL import Image
import numpy as np
from matplotlib import cm

def batch_rate_maps(study: Study, tasks: dict):
    """
    Computes rate maps across all animals, sessions, cells in a study.

    Use tasks dictionary as true/false flag with variable to compute
    e.g. {'rate_map': True, 'binary_map': False}
    """
    animals = study.animals

    for animal in animals:

        # cells, waveforms = sort_cell_spike_times(animal)

        sort_cell_spike_times(animal)
        
        c = 0
        for session in animal.session_keys:

            pos_x = animal.spatial_dict[session]['pos_x']
            pos_y = animal.spatial_dict[session]['pos_y']
            pos_t = animal.spatial_dict[session]['pos_t']
            arena_size = animal.spatial_dict[session]['arena_size']

            v = speed2D(pos_x, pos_y, pos_t)

            smoothing_factor = 5
            # Kernel size
            kernlen = int(smoothing_factor*8)
            # Standard deviation size
            std = int(0.2*kernlen)

            occupancy_map, _, _ = get_occupancy_map(pos_x, pos_y, pos_t, arena_size, kernlen, std)

            k = 0
            for cell in animal.agg_cell_keys[c]:

                spikex, spikey, spiket, _ = spikePos(animal.agg_sorted_events[c][k], pos_x, pos_y, pos_t, pos_t, False, False)

                spiket = spiket.flatten()

                rate_map_smooth, rate_map_raw = get_rate_map(pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)

                ratemap_stats_dict  = rate_map_stats(rate_map_smooth, occupancy_map)
                

                autocorr_map = get_autocorrelation(rate_map_smooth, pos_x, pos_y, arena_size)

                cell_stats = {}
                cell_stats['rate_map_smooth'] = rate_map_smooth
                cell_stats['occupancy_map'] = occupancy_map
                cell_stats['rate_map_raw'] = rate_map_raw

                if tasks['binary_map']:
                    binary_map = get_binary_map(rate_map_smooth)
                    binary_map_im = Image.fromarray(np.uint8(binary_map*255))
                    cell_stats['binary_map'] = binary_map
                    cell_stats['binary_map_im'] = binary_map_im

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
                    coherence = rate_map_coherence(rate_map_raw)
                    cell_stats['coherence'] = coherence
                    
                if tasks['speed_score']:
                    pos_t = np.array(pos_t).flatten()
                    v = np.array(v).flatten()
                    s_score = speed_score(spiket,pos_t,v)
                    cell_stats['speed_score'] = s_score

                if tasks['hd_score'] or tasks['tuning_curve']:
                    tuned_data, spike_angles, angular_occupancy, bin_array = compute_tuning_curve(pos_x, pos_y, pos_t, spiket, 2)
                    cell_stats['tuned_data'] = tuned_data
                    cell_stats['tuned_data_angles'] = spike_angles
                    cell_stats['angular_occupancy'] = angular_occupancy
                    cell_stats['angular_occupancy_bins'] = angular_occupancy

                if tasks['hd_score']:
                    hd_hist = get_hd_histogram(spike_angles)
                    cell_stats['hd_hist'] = hd_hist

                if tasks['grid_score']:
                    true_grid_score = compute_grid_score(occupancy_map, spiket, pos_x, pos_y, pos_t, arena_size, spikex, spikey, kernlen, std)
                    cell_stats['grid_score'] = true_grid_score

                if tasks['border_score']:
                    if not tasks['binary_map']:
                        binary_map = get_binary_map(rate_map_smooth)
                    b_score = compute_border_score(binary_map, rate_map_smooth)
                    cell_stats['b_score_top'] = b_score[0]
                    cell_stats['b_score_bottom'] = b_score[1]
                    cell_stats['b_score_left'] = b_score[2]
                    cell_stats['b_score_right'] = b_score[3]

                if tasks['field_sizes']:
                    image, n_labels, labels, centroids, field_sizes = get_map_blobs(rate_map_smooth)
                    cell_stats['field_size_data'] = {'image': image, 'n_labels': n_labels, 'labels': labels, 'centroids': centroids, 'field_sizes': field_sizes}
                
                animal.add_single_cell_stat(session, cell, cell_stats)
                k += 1
            
            c += 1
            


              

           





