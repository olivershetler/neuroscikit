import numpy as np

from _prototypes.wave_form_sorter.detect_peaks import detect_peaks


def get_spike_statistics(cell_waveforms, sample_rate):
    """
    Get spike width data for one cell

    input shape is (num spikes, num channel, num samples per waveform)
    """
    avg_waveform = np.mean(cell_waveforms, axis=0)

    pp_amp = np.zeros((cell_waveforms.shape[0], cell_waveforms.shape[1]))
    pt_width = np.zeros_like(pp_amp)
    peak_cell = np.zeros_like(pp_amp)

    reshaped_waveforms = np.zeros((cell_waveforms.shape[1], cell_waveforms.shape[0], cell_waveforms.shape[2]))
    for i in range(len(cell_waveforms)):
        for j in range(len(cell_waveforms[i])):
            reshaped_waveforms[j,i] = cell_waveforms[i,j]
    cell_waveforms = reshaped_waveforms

    for chan_index, channel in enumerate(cell_waveforms):

        for spike_index, spike in enumerate(channel):

            t = ((10 ** 6) / sample_rate) * np.arange(0, len(spike))

            locs = detect_peaks(spike, edge='both', threshold=0)

            pks = spike[locs]

            if len(pks) == 0:
                peak_cell[spike_index, :] = np.NaN

            else:
                max_ind = np.where(pks == np.amax(pks))[0]  # finding the index of the max value
                locs_ts = locs * (10 ** 6. / sample_rate)  # converting the peak locations to seconds
                min_val = np.amin(spike[np.where(t > locs_ts[max_ind[0]])[0]])  # finding the minimum value
                min_ind = np.where((spike == min_val) & (t > locs_ts[max_ind[0]]))[0][
                    0]  # finding the index of the minimum value
                min_t = t[min_ind]  # converting the index to a time value
                peak_cell[spike_index, :] = np.array([np.amax(pks), locs_ts[max_ind[0]], min_val, min_t])


            pp_amp[:, chan_index] = peak_cell[:, 0] - peak_cell[:, 2]
            pt_width[:, chan_index] = peak_cell[:, 1] - peak_cell[:, 3]

            # ------------ Done calculating the peak to peaks and peak to through values ------------- #

        avg_pp_amp = np.zeros((1, len(cell_waveforms)))
        avg_pt_width = np.zeros_like(avg_pp_amp)

        for waveform_index, waveform in enumerate(avg_waveform):
            locs = detect_peaks(waveform, edge='both', threshold=0)
            locs_ts = locs * (10 ** 6. / sample_rate)  # converting the peak locations to seconds
            pks = waveform[locs]
            max_val = np.amax(pks)
            max_ind = np.where(pks == max_val)[0]  # finding the index of the max value
            max_t = locs_ts[max_ind[0]]
            min_val = np.amin(waveform[np.where(t > locs_ts[max_ind[0]])[0]])  # finding the minimum value
            min_ind = np.where(waveform == min_val)[0]  # finding the index of the minimum value
            min_t = t[min_ind]  # converting the index to a time value
            avg_pp_amp[0, waveform_index] = max_val - min_val
            avg_pt_width[0, waveform_index] = min_t - max_t

    # ------------ ended avg_waveform calculations ------------------- #
    best_channel = np.where(avg_pp_amp == np.amax(avg_pp_amp))[1][0]

    channel = 'ch%d' % int(best_channel + 1)
    waveform_dict = {}
    for channel_index, channel_waveform in enumerate(cell_waveforms):
        waveform_dict['ch%d' % int(channel_index + 1)] = channel_waveform
    waveform_dict.update({'sample_rate': sample_rate, 'pp_amp': avg_pp_amp[0],
                          'spike_width': avg_pt_width[0], 'peak_channel': best_channel})
    return waveform_dict