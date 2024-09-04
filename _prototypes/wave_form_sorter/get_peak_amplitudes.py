import numpy as np

def get_peak_amplitudes(waveforms):
    """
    Get peak amplitudes and ids of peaks
    """
    peaks = [[] for i in range(len(waveforms))]
    peak_ids = [[] for i in range(len(waveforms))]
    for i in range(len(waveforms)):
        for j in range(len(waveforms[i])):
            cell_peak = []
            cell_peak_id = []
            for k in range(len(waveforms[i][j])):
                cell_peak.append(np.max(waveforms[i][j][k]))
                cell_peak_id.append(np.argmax(waveforms[i][j][k]))
            peaks[i].append(cell_peak)
            peak_ids[i].append(cell_peak_id)
    return peaks, peak_ids