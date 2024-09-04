import numpy as np
import itertools


def match_waveforms_by_session(agg_waveform_dict: dict):
    # Check if possible orders for each cell match across sessions

    sessions = list(agg_waveform_dict.keys())
    agg_matched = []

    for i in range(len(sessions)-1):
        session_1 = agg_waveform_dict[sessions[i]]
        session_2 = agg_waveform_dict[sessions[i+1]]
        matched = []
        for i in session_1:
            for j in session_2:
                cell_1_orders = session_1[i]['channel_orders']
                cell_1_width = session_1[i]['spike_width']
                cell_2_orders = session_2[j]['channel_orders']
                cell_2_width = session_2[j]['spike_width']
                cell_1_type = classify_cell_by_width(cell_1_width)
                cell_2_type = classify_cell_by_width(cell_2_width)
                if cell_type_match(cell_1_type, cell_2_type):
                    if len(cell_1_orders) > 0 and len(cell_2_orders) > 0:
                        assert type(cell_1_orders[0]) == list and type(cell_2_orders[0]) == list
                        for k in range(len(cell_1_orders)):
                            if cell_1_orders[k] in cell_2_orders:
                                matched.append([int(i.split('_')[-1]), int(j.split('_')[-1])])
        agg_matched.append(matched)

    matched_pairs = []
    for matched in agg_matched:
        if len(matched) ==2:
            matched_pairs.append(matched)
        if len(matched) > 2:
            for pair in itertools.combinations(matched, 2):
                waveform_overlap = waveform_overlap(pair[0], pair[1])


    return matched_pairs


def cell_type_match(cell_1_type, cell_2_type):
    if cell_1_type == cell_2_type:
        return True
    else:
        return False

def classify_cell_by_width(spike_width):
    assert spike_width > 0, 'Spike width must be greater than 0'
    if spike_width > 300:
        spike_width = 'excitatory'
    else:
        spike_width = 'inhibitory'