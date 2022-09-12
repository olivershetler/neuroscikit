import numpy as np 

def average_cell_waveforms(waveforms):
    """
    Averages cell waveforms for each session
    """
    averaged_waveforms = [[] for i in range(len(waveforms))]
    # for each session
    for i in range(len(waveforms)):
        # for each cell
        for j in range(len(waveforms[i])):
            avg = np.mean(waveforms[i][j].squeeze(), axis=0)
            averaged_waveforms[i].append(avg)
    return averaged_waveforms