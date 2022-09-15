import os
import sys
from library.spike.avg_spike_burst import avg_spike_burst

from library.spike.find_burst import find_burst

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

from core.data_study import Study, Animal

from library.cluster import create_features, L_ratio, isolation_distance
from library.spike import sort_cell_spike_times, histogram_ISI, find_burst, avg_spike_burst
from library.spatial import speed2D

from opexebo.analysis import rate_map_stats, rate_map_coherence, speed_score
from PIL import Image
import numpy as np
from matplotlib import cm

def batch_spike_analysis(study: Study):
    """
    Computes rate maps across all animals, sessions, cells in a study.

    Use tasks dictionary as true/false flag with variable to compute
    e.g. {'rate_map': True, 'binary_map': False}
    """
    animals = study.animals

    for animal in animals:

        sort_cell_spike_times(animal)
        
        c = 0
        for session in animal.session_keys:

            pos_x = animal.spatial_dict[session]['pos_x']
            pos_y = animal.spatial_dict[session]['pos_y']
            pos_t = animal.spatial_dict[session]['pos_t']

            ch1, ch2, ch3, ch4 = animal.agg_waveforms[c]
            data_concat = np.vstack((ch1, ch2, ch3, ch4)).reshape((4, -1, ch1.shape[1]))
            FD = create_features(data_concat)

            # FD = np.array(FD)

            k = 0
            for cell in animal.agg_cell_keys[c]:
                bursts, singleSpikes = find_burst(animal.agg_sorted_events[c][k])
                bursting = 100 * len(bursts) / (len(bursts) + len(singleSpikes))

                bursts_n_spikes_avg = avg_spike_burst(animal.agg_sorted_events[c][k], bursts, singleSpikes)  # num of spikes on avg per burst

                ISI_dict = histogram_ISI(animal.agg_sorted_events[c][k], FD, animal.agg_cluster_labels[c], cell)
                # cluster_spikes = np.where(animal.agg_cluster_labels[c] == int(cell))[0]  # indices of the unit we are analyzing
                cluster_spikes = animal.agg_sorted_labels[c]
                L, Lratio, df = L_ratio(FD, cluster_spikes)
                IsoDist = isolation_distance(FD, cluster_spikes)
                cluster_quality_dict = {'L': L, 'Lratio': Lratio, 'IsoDist': IsoDist}

                cell_stats = {}
                cell_stats['bursting'] = bursting
                cell_stats['bursts_n_spikes_avg'] = bursts_n_spikes_avg
                cell_stats['ISI'] = ISI_dict
                cell_stats['ClusterQuality'] = cluster_quality_dict

                animal.add_single_cell_stat(session, cell, cell_stats)
                k += 1
            
            c += 1
            


              

           





