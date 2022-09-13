import numpy as np

def get_spike_pos(cell, pos_x, pos_y, pos_t):
    """
    cell: spike times
    pos_x, pos_y: x and y position of animal in arena
    pos_t: time at (x,y) position
    """

    spike_x, spike_y = [], []

    for i in range(len(cell)):
        id = np.where(cell[i] == pos_t)[0]
        spike_x.append(pos_x[id])
        spike_y.append(pos_y[id])

    return spike_x, spike_y

