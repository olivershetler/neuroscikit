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
    waveforms = clusters.waveforms

    assert len(spike_times) == len(waveforms[0])

    cells = []
    sorted_waveforms = []
    good_cells = []
    good_sorted_waveforms = []
    good_sorted_label_ids = []

    labels = np.unique(cluster_labels)
    # comes in shape (channel count, spike time, nmb samples) but is nested list not numpy
    # want to rearrannge to be (spike time, channel count, nmb sample)
    waves = np.array(waveforms).reshape((len(waveforms[0]), len(waveforms),  len(waveforms[0][0])))
    for lbl in labels:
        idx = np.where(cluster_labels == lbl)
        cells.append(np.array(spike_times)[idx])
        sorted_waveforms.append(waves[idx,:,:].squeeze())

    empty_cell = 1
    for j in range(len(sorted_waveforms)):
        if len(sorted_waveforms[j]) == 0 and j != 0:
            empty_cell = j
            break
        else:
            empty_cell = j + 1
    for j in range(1,empty_cell,1):
        good_cells.append(cells[j])
        good_sorted_label_ids.append(j)
        good_sorted_waveforms.append(sorted_waveforms[j])


    return good_cells, good_sorted_waveforms
