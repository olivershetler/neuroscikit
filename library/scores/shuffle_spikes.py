import numpy as np
from opexebo.general import shuffle

def shuffle_spikes(self, ts: np.ndarray, pos_x: np.ndarray,
                   pos_y: np.ndarray, pos_t: np.ndarray) -> list:

    '''
        Shuffles spike and position data.

        Params:
            ts (np.ndarray):
                Timestamps of spike events
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y coordinates as well as position timestamps

        Returns:
            np.ndarray: shuffled_spikes
            --------
            shuffled_spikes:
                Array where columns are shuffled x and y spike coordinates respectively.
    '''

    spike_x = np.zeros((1,len(ts)))
    spike_y = np.zeros((1,len(ts)))
    shuffled_spike_xy = np.zeros((2,len(ts)))
    # Compute a shuffling between 20 and 100

    shuffled_times = shuffle(ts, 20, 100, t_start=min(ts), t_stop = max(ts))[0]

    shuffled_spikes = []
    # For each shuffled time, find and place the corresponding
    # spike x and y coordinates into the shuffled_spikes array
    for i, single_shuffled_time in enumerate(shuffled_times):
        for j, time in enumerate(single_shuffled_time):
            index = np.abs(pos_t - time).argmin()
            spike_x[0][j] = pos_x[index]
            spike_y[0][j] = pos_y[index]

        # Emit shuffle progress for progress bar in ShuffleWindow
        self.signals.progress.emit(i)

        shuffled_spike_xy[0] = spike_x
        shuffled_spike_xy[1] = spike_y
        shuffled_spikes.append(shuffled_spike_xy.copy())

    return shuffled_spikes
