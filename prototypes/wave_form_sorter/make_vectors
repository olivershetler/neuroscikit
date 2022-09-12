import numpy as np

def make_vectors(agg_waveform_dict: dict):
    """
    Vectors will be in format [[start x, start y], [end x, end y]]
    
    """

    agg_spike_amp, agg_spike_width, agg_peak_channel = aggregate_amp_width(agg_waveform_dict)

    vectors = [[] for i in range(len(spike_amps))]

    tet_x_coord = [-1,1,1,-1]
    tet_y_coord = [1,1,-1,-1]

    for i in range(len(spike_amps)):
        ses_amps = spike_amps[i]
        mn = np.mean(ses_amps)
        std = np.std(ses_amps)
        for j in range(len(ses_amps)):
            cell_amps = (ses_amps[j] - mn)/std
            cell_widths = spike_widths[i][j]
            cell_peak = peak_channels[i][j]

            loc_x = np.sum(np.array(tet_x_coord) * np.array(cell_amps))
            loc_y = np.sum(np.array(tet_y_coord) * np.array(cell_amps))

            vector = [loc_x, loc_y, tet_x_coord[cell_peak], tet_y_coord[cell_peak]]
            # vector = [0, 0, loc_x, loc_y]
            vectors[i].append(vector)
            
    return vectors

def aggregate_amp_width(agg_waveform_dict: dict):
    # Aggregate spike width and amp

    agg_spike_amp = [[] for i in range(len(agg_waveform_dict))]
    agg_spike_width = [[] for i in range(len(agg_waveform_dict))]
    agg_peak_channel = [[] for i in range(len(agg_waveform_dict))]

    c = 0
    for session_key in agg_waveform_dict:
        session = agg_waveform_dict[session_key]
        for cell_key in session:
            cell = session[cell_key]
            agg_spike_amp[c].append(cell['pp_amp'])
            agg_spike_width[c].append(cell['spike_width'])
            agg_peak_channel[c].append(cell['peak_channel'])
        c += 1

    agg_spike_amp = np.array(agg_spike_amp)
    agg_spike_width = np.array(agg_spike_width)
    agg_peak_channel = np.array(agg_peak_channel)

    return agg_spike_amp, agg_spike_width, agg_peak_channel