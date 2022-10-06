import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

import numpy as np
import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.batch_space import SpikeClusterBatch

def sort_spikes_by_cell(clusters: SpikeClusterBatch):
    # spike_times, cluster_labels, waveform
    """
    Takes multiple sessions with spike times, cluster_labels and waveforms.

    Returns valid cells for each session and associated waveforms
    """

    spike_times = clusters.event_times
    cluster_labels = clusters.cluster_labels
    waveforms = clusters.get_all_channel_waveforms()

    assert len(spike_times) == len(waveforms[0])

    cells = []
    sorted_waveforms = []
    good_cells = []
    good_sorted_waveforms = []
    good_sorted_label_ids = []
    good_clusters = []

    labels = np.unique(cluster_labels)
    # comes in shape (channel count, spike time, nmb samples) but is nested list not numpy
    # want to rearrannge to be (spike time, channel count, nmb sample)
    waves = np.array(waveforms).reshape((len(waveforms[0]), len(waveforms),  len(waveforms[0][0])))
    for lbl in labels:
        idx = np.where(cluster_labels == lbl)
        cells.append(np.array(spike_times)[idx])
        sorted_waveforms.append(waves[idx,:,:].squeeze())
        # sorted_clusters.append(indiv_clusters[idx])

    empty_cell = 1
    for j in range(len(sorted_waveforms)):
        if len(sorted_waveforms[j]) == 0 and j != 0:
            empty_cell = j
            break
        else:
            empty_cell = j + 1

    unique_labels = np.asarray(clusters.get_unique_cluster_labels())
    idx = np.where((unique_labels >= 1) & (unique_labels < empty_cell))
    good_sorted_label_ids = unique_labels[idx]
    clusters.set_sorted_label_ids(good_sorted_label_ids)
    indiv_clusters = clusters.get_spike_cluster_instances()

    for j in good_sorted_label_ids:
        good_cells.append(cells[j])
        good_sorted_waveforms.append(sorted_waveforms[j])
        # indiv_clusters will only be made for good_label_id cells so have to adjust to make 0 index when pulling out spike cluster
        good_clusters.append(indiv_clusters[j-1])
        assert indiv_clusters[j-1].cluster_label == j
    
    

    return good_cells, good_sorted_waveforms, good_clusters, good_sorted_label_ids
