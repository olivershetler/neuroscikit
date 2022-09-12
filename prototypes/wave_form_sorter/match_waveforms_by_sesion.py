import numpy as np 


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
                cell_1 = session_1[i]['channel_orders']
                cell_2 = session_2[j]['channel_orders']
                if len(cell_1) > 0 and len(cell_2) > 0:
                    if type(cell_1[0]) == list and type(cell_2[0]) == list:
                        for k in range(len(cell_1)):
                            if cell_1[k] in cell_2:
                                matched.append([int(i.split('_')[-1]), int(j.split('_')[-1])])
                    elif type(cell_1[0]) == list and type(cell_2[0]) != list:
                        for k in range(len(cell_1)):
                            if cell_1[k] == cell_2:
                                matched.append([int(i.split('_')[-1]), int(j.split('_')[-1])])
                    elif type(cell_1[0]) != list and type(cell_2[0]) == list:
                        for k in range(len(cell_2)):
                            if cell_1 == cell_2[k]:
                                matched.append([int(i.split('_')[-1]), int(j.split('_')[-1])])
                    elif type(cell_1[0]) != list and type(cell_2[0]) != list:
                        if cell_1 == cell_2:
                            matched.append([int(i.split('_')[-1]), int(j.split('_')[-1])])
        agg_matched.append(matched)

    return agg_matched