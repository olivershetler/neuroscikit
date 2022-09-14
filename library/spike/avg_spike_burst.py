import numpy as np

def avg_spike_burst(ts, bursts, singleSpikes):
    """calculates the average spikes per burst"""
    total_n = np.arange(len(ts.flatten()))
    burst_spikes = np.setdiff1d(total_n, singleSpikes)  # these are all indices belonging to the bursts

    # divide by total number of burst events to return the output

    if len(bursts) != 0:
        return len(burst_spikes) / len(bursts)
    else:
        return np.NaN
