import numpy as np

def waveform_template(waveforms):
    """
    Averages cell waveforms for each session
    """
    averaged_waveforms = [[] for i in range(len(waveforms))]
    upper_bound_waveform = [[] for i in range(len(waveforms))]
    lower_bound_waveform = [[] for i in range(len(waveforms))]
    # for each session
    for i in range(len(waveforms)):
        # for each cell
        for j in range(len(waveforms[i])):
            avg = np.mean(waveforms[i][j].squeeze(), axis=0)
            averaged_waveforms[i].append(avg)
            upper_bound = np.quantile(avg, 0.8)
            upper_bound_waveform[i].append(upper_bound)
            lower_bound = np.quantile(avg, 0.2)
            lower_bound_waveform[i].append(lower_bound)
    return (averaged_waveforms, upper_bound_waveform, lower_bound_waveform)