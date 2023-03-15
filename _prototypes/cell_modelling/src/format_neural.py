import numpy as np
import pandas as pd

def event_times_to_count(event_times, T, dt):
    # dt = settings_dict['bin_timestep']
    new_time_index = np.arange(0,T+dt,dt)
    ct, bins = np.histogram(event_times, bins=new_time_index)
    ct = np.asarray(ct)
    # norm_ct = (ct/dt - np.mean(ct/dt))/np.std(ct/dt)
    # norm_ct[norm_ct < 0] = 0 # cant have negative count

    return ct/dt, bins[:-1]
    # CHECK IF NEED NORMALIZE
    # return np.array(ct)/dt, bins[:-1]

def aggregate_spike_objects(animal):
    max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

    def _get_single_cell_object(k,i,animal):
        if k+1 in animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_label_dict():
            return animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_by_id(k+1)

    def _get_valid_cell_object(k, animal):
        return list(map(lambda i: _get_single_cell_object(k, i+1, animal), np.arange(len(list(animal.sessions.keys())))))
    
    spike_objects = list(map(lambda k: _get_valid_cell_object(k, animal), np.arange(int(max_matched_cell_count))))
    spike_objects = spike_objects[not np.isnan(spike_objects)]

    return spike_objects

def aggregate_spike_times(animal):
    max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

    def _get_single_cell_spike_times(k,i,animal):
        if k+1 in animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_label_dict():
            return animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_by_id(k+1).event_times
        else:
            return np.nan
    
    def _get_valid_spike_times(k, animal):
        spk_times = np.array(list(map(lambda i: _get_single_cell_spike_times(k, i+1, animal), np.arange(len(list(animal.sessions.keys()))))))
        if len(spk_times.shape) == 2:
            ses_ct, spk_ct = spk_times.shape
            spk_times = spk_times[~np.isnan(spk_times)]
            print('reshaping spk time with shape {} to shape ({},{})'.format(spk_times.shape, ses_ct, spk_ct))
            new_spk_times = spk_times.reshape((ses_ct, spk_ct))
        else:
            # new_spk_times = list(map(lambda x: spk_times[x][pd.isnull(spk_times[x])], np.arange(len(spk_times))))
            new_spk_times = spk_times[~pd.isnull(spk_times)]

        return new_spk_times
    
    spike_times = list(map(lambda k: _get_valid_spike_times(k, animal), np.arange(int(max_matched_cell_count))))

    return spike_times

def aggregate_spike_counts(animal, settings_dict):
    max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

    def _get_single_cell_spike_cts(k,i,animal):
        if k+1 in animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_label_dict():
            cell = animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_by_id(k+1)
            ct, _ = event_times_to_count(cell.event_times, cell.cluster.time_index[-1], settings_dict['bin_timestep'])
            return ct
    
    def _get_valid_spike_cts(k, animal):
        spk_counts = np.array(list(map(lambda i: _get_single_cell_spike_cts(k, i+1, animal), np.arange(len(list(animal.sessions.keys()))))))
        spk_counts = spk_counts[~np.isnan(spk_counts)]
        return spk_counts
    
    spike_counts = list(map(lambda k: _get_valid_spike_cts(k, animal), np.arange(int(max_matched_cell_count))))

    return spike_counts

def aggregate_trial_types(animal, settings_dict):
    max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

    def _get_single_cell_trial_types(k,i,animal):
        if k+1 in animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_label_dict():
            cell = animal.sessions['session_' + str(i)].get_cell_data()['cell_ensemble'].get_cell_by_id(k+1)
            trial_order = list(cell.cluster.session_metadata.file_paths['tet'].split('-')[-1].split('.')[0].split('odor')[-1])
            if settings_dict['firstIsBaseline']:
                trial_order = np.hstack((['o'], trial_order))
            return trial_order
    
    def _get_valid_trial_types(k, animal):
        ttypes = np.array(list(map(lambda i: _get_single_cell_trial_types(k, i+1, animal), np.arange(len(list(animal.sessions.keys()))))))
        if len(ttypes.shape) == 2:
            ses_ct, ttype_ct = ttypes.shape
            ttypes = ttypes[~pd.isnull(ttypes)]
            print('reshaping spk time with shape {} to shape ({},{})'.format(ttypes.shape, ses_ct, ttype_ct))
            new_ttypes = ttypes.reshape((ses_ct, ttype_ct))
        else:
            new_ttypes = ttypes[~pd.isnull(ttypes)]
        return new_ttypes
    
    trial_types = list(map(lambda k: _get_valid_trial_types(k, animal), np.arange(int(max_matched_cell_count))))

    return trial_types

# helper fxn
def _filtEvent(x, event_time, start, end):
    if event_time>= start and event_time < end:
        return x
    else:
        return np.nan
    
    
def _concat_single_session_trials(i, j, agg_events, agg_trial_types, cell_trials, cell_trial_labels, prev_trial, prev_trial_id, settings):
    trial_start_times, trial_length, dt = settings['trial_start_times'], settings['trial_length'], settings['bin_timestep']

    ses_events = agg_events[i][j]

    # for one of the possible trials
    for k in range(len(trial_start_times)):
        start = trial_start_times[k]
        end = trial_start_times[k] + trial_length
        # trial_id = trial_ids[k]

        # get events only in trial window
        ids = np.array(list(map(lambda x: _filtEvent(x,ses_events[x],start,end), np.arange(0, len(ses_events), 1))))
        ids = ids[ids == ids]

        # convert to binned counts
        ct, new_time_index = event_times_to_count(np.array(agg_events[i][j])[np.array(ids, dtype=np.int32)] - start, trial_length, dt)
        # ct = ct/np.sum(ct)

        # trial id e.g. odor id, object id, ...
        trial_id = agg_trial_types[i][j][k]  

        # concatenate
        cell_trials = np.hstack((cell_trials, ct))
        cell_trial_labels = np.hstack((cell_trial_labels, [trial_id]))

        if settings['addInterTrial']:
            if prev_trial_id is not None:
                if trial_id != 'o' and k != len(trial_start_times)-1:
                    cell_trials = np.hstack((cell_trials, prev_trial))
                    cell_trial_labels = np.hstack((cell_trial_labels, [prev_trial_id]))
                else:
                    prev_trial_id = trial_id
                    prev_trial = ct
            else:
                prev_trial_id = trial_id
                prev_trial = ct

    return {'cell_trials': cell_trials, 'cell_trial_labels': cell_trial_labels, 'prev_trial':prev_trial, 'prev_trial_id':prev_trial_id, 'new_time_index':new_time_index}
    
def _concat_cell_trials(i, agg_events, agg_trial_types, new_time_index, settings):

    cell_trials = []
    cell_trial_labels = []
    prev_trial_id = None
    prev_trial = None

    # print(list(map(lambda j: _concat_single_session_trials(i, j, agg_events, agg_trial_types, cell_trials, cell_trial_labels, prev_trial, prev_trial_id, settings), np.arange(len(agg_events[i])))))
    # cell_trials, cell_trial_labels, prev_trial, prev_trial_id, new_time_index = [list(map(lambda j: _concat_single_session_trials(i, j, agg_events, agg_trial_types, cell_trials, cell_trial_labels, prev_trial, prev_trial_id, settings), np.arange(len(agg_events[i])))) for k in keys]
    # # cell_trials, cell_trial_labels, prev_trial, prev_trial_id, new_time_index = res[:,0], res[:,1], res[:,2], res[:,3], res[:,4]

    trial_start_times, trial_length, dt = settings['trial_start_times'], settings['trial_length'], settings['bin_timestep']

    for j in range(len(agg_events[i])):

        ses_events = agg_events[i][j]
        cell_session_trials = []
        cell_session_trial_labels = []
                   
        # for one of the possible trials
        for k in range(len(trial_start_times)):
            start = trial_start_times[k]
            end = trial_start_times[k] + trial_length
            # trial_id = trial_ids[k]

            # get events only in trial window
            ids = np.array(list(map(lambda x: _filtEvent(x,ses_events[x],start,end), np.arange(0, len(ses_events), 1))))
            ids = ids[ids == ids]

            # convert to binned counts
            ct, new_time_index = event_times_to_count(np.array(agg_events[i][j])[np.array(ids, dtype=np.int32)] - start, trial_length, dt)
            # ct = ct/np.sum(ct)

            # trial id e.g. odor id, object id, ...
            trial_id = agg_trial_types[i][j][k]  

            # concatenate
            cell_session_trials = np.hstack((cell_session_trials, ct))
            cell_session_trial_labels = np.hstack((cell_session_trial_labels, [trial_id]))

            if settings['addInterTrial']:
                if prev_trial_id is not None:
                    if trial_id != 'o' and k != len(trial_start_times)-1:
                        cell_session_trials = np.hstack((cell_session_trials, prev_trial))
                        cell_session_trial_labels = np.hstack((cell_session_trial_labels, [prev_trial_id]))
                    else:
                        prev_trial_id = trial_id
                        prev_trial = ct
                else:
                    prev_trial_id = trial_id
                    prev_trial = ct

        cell_trials.append(cell_session_trials)
        cell_trial_labels.append(cell_session_trial_labels)

    return cell_trials, cell_trial_labels, new_time_index
            
def sequence_cell_trials(agg_events, agg_trial_types, settings):
    new_time_index = None

    # sequential trials will be concatenated sessions for a given matched cell
    sequential_trials = []
    sequential_labels = []
    # for cell in aggregated events (cell,session,events)

    sequential_trials, sequential_labels, new_time_index = list(np.array(list(map(lambda i: _concat_cell_trials(i, agg_events, agg_trial_types, new_time_index, settings), np.arange(len(agg_events))))).T)
    # sequential_trials, sequential_labels, new_time_index = res[:,0], res[:,1], res[:,2]

    return np.asarray(sequential_trials), np.asarray(sequential_labels), np.asarray(new_time_index)
