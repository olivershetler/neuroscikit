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
    Returns valid cells for session and associated waveforms
    """

    spike_times = clusters.event_times
    cluster_labels = clusters.cluster_labels
    waveforms = clusters.get_all_channel_waveforms()

    assert len(spike_times) == len(waveforms[0])

    cells = []
    sorted_waveforms = []
    good_cells = []
    good_sorted_waveforms = []
    sorted_label_ids = []
    good_clusters = []

    unique_labels = np.unique(cluster_labels)
    # comes in shape (channel count, spike time, nmb samples) but is nested list not numpy
    # want to rearrannge to be (spike time, channel count, nmb sample)
    # waves = np.array(waveforms).reshape((len(waveforms[0]), len(waveforms),  len(waveforms[0][0])))
    waves = np.swapaxes(waveforms, 1, 0)
    # waves = np.asarray(waveforms)
    for lbl in unique_labels:
        idx = np.where(cluster_labels == lbl)[0]
        idx = idx[idx <= len(spike_times)-1]
        # print(np.array(spike_times).squeeze().shape, idx)
        spks = np.array(spike_times).squeeze()[idx]
        if type(spks) == float or type(spks) == np.float64:
            spks = [spks]
        if len(spks) < 40000 and len(spks) > 100:
            cells.append(spks)
            sorted_waveforms.append(waves[idx,:,:].squeeze())
            # sorted_waveforms.append(waves[:,idx,:].squeeze())
            sorted_label_ids.append(lbl)
        # sorted_clusters.append(indiv_clusters[idx])

    # empty_cell = 0
    # for j in range(len(sorted_waveforms)):
    #     print(j, len(sorted_waveforms[j]), empty_cell)
    #     if len(sorted_waveforms[j]) == 0 and j != 0:
    #         empty_cell = j
    #         break
    #     else:
    #         empty_cell = j + 1

    # print(unique_labels)
    # empty_cell = sorted(set(range(unique_labels[0], unique_labels[-1] + 1)).difference(unique_labels))

    if unique_labels[0] == 0:
        empty_cell = sorted(set(range(unique_labels[1], unique_labels[-1] + 1)).difference(unique_labels))
    else:
        empty_cell = sorted(set(range(unique_labels[0], unique_labels[-1] + 1)).difference(unique_labels))

    if len(empty_cell) >= 1:
        empty_cell = empty_cell[0]
    else:
        empty_cell = unique_labels[-1] + 1

    sorted_label_ids = np.asarray(sorted_label_ids)
    idx = np.where((sorted_label_ids >= 1) & (sorted_label_ids < empty_cell))
    good_sorted_label_ids = sorted_label_ids[idx]

    # VERY IMPORTANT LINE #
    clusters.set_sorted_label_ids(good_sorted_label_ids)
    # IF SORTED LABEL IDS NOT SET, NOISE LABELS WILL BE USED TO MKAE CELL #

    indiv_clusters = clusters.get_spike_cluster_instances()
    # print(len(indiv_clusters), len(sorted_label_ids), sorted_label_ids, empty_cell)
    # good_label_ids = []
    # for j in sorted_label_ids:
    for j in range(len(good_sorted_label_ids)):
        label_id = good_sorted_label_ids[j]
        # print(len(cells[j]))
        good_cells.append(cells[j])
        good_sorted_waveforms.append(sorted_waveforms[j])
        # indiv_clusters will only be made for good_label_id cells so have to adjust to make 0 index when pulling out spike cluster
        good_clusters.append(indiv_clusters[j])
        assert indiv_clusters[j].cluster_label == label_id



    return good_cells, good_sorted_waveforms, good_clusters, good_sorted_label_ids
