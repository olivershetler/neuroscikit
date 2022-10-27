import numpy as np
from _prototypes.unit_matcher.waveform import time_index, troughs

def extract_average_spike_widths(study) -> dict:
    output_df = {'session_signature':[], 'tetrode':[], 'unit_id':[], 'spike_width':[], 'firing_rate':[]}
    for animal in study.animals:
        for key, session in animal.sessions.items():
            cluster_labels = session.session_data.data['spike_cluster'].get_unique_cluster_labels()
            session_signature = session.session_metadata.file_paths['pos'][:-4].split('\\')[-1].split('/')[-1]
            tetrode = session.session_metadata.file_paths['tet'].split('.')[-1]
            for unit in cluster_labels:
                n_spikes, spike_times, waveforms = session.session_data.data['spike_cluster'].get_single_spike_cluster_instance(unit)
                sample_rate = session.session_data.data['spike_cluster'].sample_rate
                time_step = 1 / sample_rate

                wf_avg = np.array(waveforms).mean(axis=1)
                max_vals = list(map(lambda x: max(x), [wf_avg[0], wf_avg[1], wf_avg[2], wf_avg[3]]))
                principal_channel_index = np.argmax(max_vals)
                principal_waveform = wf_avg[principal_channel_index]
                peak_index = np.argmax(principal_waveform)
                trough_list = list(filter(lambda x: x > peak_index, troughs(principal_waveform, time_step)))
                if len(trough_list) > 0:
                    trough_index = trough_list[0]
                    spike_width = (trough_index - peak_index) * time_step
                else:
                    trough_index = len(principal_waveform) - 1
                    spike_width = int((trough_index - peak_index)/2) * time_step
                assert spike_width > 0, 'Spike width is negative'
                firing_rate = n_spikes/spike_times[-1]
                output_df['session_signature'].append(session_signature)
                output_df['tetrode'].append(tetrode)
                output_df['unit_id'].append(unit)
                output_df['spike_width'].append(spike_width)
                output_df['firing_rate'].append(firing_rate)
    return output_df