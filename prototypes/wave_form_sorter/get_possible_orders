import numpy as np

def get_switch_channels(pp_amp, threshold):
    """
    For session for cell, get channel in tetrode that could switch in a possible ordering
    """
    can_switch = []
    for i in range(len(pp_amp)):
        for j in range(len(pp_amp)):
            pair = np.sort([i,j]).tolist()
            if i != j and pair not in can_switch:
                diff = abs(pp_amp[i] - pp_amp[j])
                if diff <= threshold:
                    can_switch.append(pair)
    return can_switch

def get_possible_orders(pp_amp, threshold=.2):
    """ 
    Get possible ordering for a cell
    """
    # avg_waveform = np.mean(cell_waveforms, axis=0)
    possible_orders = []

    can_switch = get_switch_channels(pp_amp, threshold)
    switched = []
    true_order = np.argsort(pp_amp)
    possible_orders.append(true_order.tolist())
    
    for i in range(len(pp_amp)):
        for j in range(len(can_switch)):
            if i in can_switch[j] and can_switch[j] not in switched:
                new_order = np.copy(true_order)
                id1 = np.where(new_order == can_switch[j][0])[0]
                id2 = np.where(new_order == can_switch[j][1])[0]
                new_order[id1] = can_switch[j][1]
                new_order[id2] = can_switch[j][0]
                switched.append(can_switch[j])
                possible_orders.append(new_order.tolist())

    return possible_orders