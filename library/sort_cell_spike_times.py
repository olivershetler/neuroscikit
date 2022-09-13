import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

import numpy as np

from core.data_study import Animal

def sort_cell_spike_times(animal: Animal):
    # spike_times, cluster_labels, waveform
    """
    Takes multiple sessions with spike times, cluster_labels and waveforms.

    Returns valid cells for each session and associated waveforms
    """

    spike_times = animal.agg_spike_times
    cluster_labels = animal.agg_cluster_labels
    waveforms = animal.agg_sorted_waveforms

    assert len(spike_times) == len(cluster_labels)

    cells = [[] for i in range(len(spike_times))]
    sorted_waveforms = [[] for i in range(len(spike_times))]
    good_cells = [[] for i in range(len(spike_times))]
    good_sorted_waveforms = [[] for i in range(len(spike_times))]

    for i in range(len(spike_times)):
        labels = np.unique(cluster_labels[i])
        # comes in shape (channel count, spike time, nmb samples) but is nested list not numpy
        # want to rearrannge to be (spike time, channel count, nmb sample )
        waves = np.array(waveforms[i]).reshape((len(waveforms[i][0]), len(waveforms[i]),  len(waveforms[i][0][0])))
        for lbl in labels:
            idx = np.where(cluster_labels[i] == lbl)
            cells[i].append(np.array(spike_times[i])[idx])
            sorted_waveforms[i].append(waves[idx,:,:].squeeze())

        empty_cell = 1
        for j in range(len(sorted_waveforms[i])):
            if len(sorted_waveforms[i][j]) == 0 and j != 0:
                empty_cell = j
                break
            else:
                empty_cell = j + 1
        for j in range(1,empty_cell,1):
            good_cells[i].append(cells[i][j])
            good_sorted_waveforms[i].append(sorted_waveforms[i][j])

    animal.add_sorted_data(good_cells, good_sorted_waveforms)
    
    # return good_cells, good_sorted_waveforms
