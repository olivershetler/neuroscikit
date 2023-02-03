import numpy as np
import os,sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)



def event_times_to_count(event_times, T, dt):
    # dt = settings_dict['bin_timestep']
    new_time_index = np.arange(0,T,dt)
    ct, bins = np.histogram(event_times, bins=new_time_index)
    # print(ct.shape, bins[:-1].shape)
    return np.array(ct)/dt, bins[:-1]

def sequence_cell_trials(agg_events, agg_event_objects, agg_trial_types, settings):
    trial_start_times, trial_length, dt = settings['trial_start_times'], settings['trial_length'], settings['bin_timestep']

    # helper fxn
    def _filtEvent(x, event_time, start, end):
        if event_time>= start and event_time < end:
            return x

    # def _sequenceEvents()
    
    # sequential trials will be concatenated sessions for a given matched cell
    sequential_trials = []
    sequential_labels = []
    trial_dict = {}
    # for cell in aggregated events (cell,session,events)
    for i in range(len(agg_events)):
        trial_dict[i] = {}
        trial_dict[i]['obj'] = {}
        trial_dict[i]['data'] = {}

        cell_trials = []
        cell_trial_labels = []
        prev_trial_id = None
        prev_trial = None

        # for ses that has cell
        for j in range(len(agg_events[i])):
            ses_events = agg_events[i][j]

            # for one of the possible trials
            for k in range(len(trial_start_times)):
                start = trial_start_times[k]
                end = trial_start_times[k] + trial_length
                # trial_id = trial_ids[k]

                # get events only in trial window
                ids = np.array(list(map(lambda x: _filtEvent(x,ses_events[x],start,end), np.arange(0, len(ses_events), 1))))
                ids = ids[ids != None]

                # convert to binned counts
                ct, new_time_index = event_times_to_count(np.array(agg_events[i][j])[np.array(ids, dtype=np.int32)] - start, trial_length, dt)
                # ct = ct/np.sum(ct)

                # cell object
                obj = agg_event_objects[i][j]  

                # trial id e.g. odor id, object id, ...
                trial_id = agg_trial_types[i][j][k]  

                print(trial_id, len(cell_trial_labels)) 

                if trial_id not in trial_dict[i]['data']:
                    trial_dict[i]['data'][trial_id] = []
                    trial_dict[i]['obj'][trial_id] = []  

                trial_dict[i]['data'][trial_id].append(ct)
                trial_dict[i]['obj'][trial_id].append(obj)
                
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

        # array of concatenated cells
        sequential_trials.append(cell_trials)
        sequential_labels.append(cell_trial_labels)

    # trial dict stores cell object and individual trial data, can build separate analyses e.g. only odor X or odor A trials using this dict to extract arrays
    # new time index from binned spk counts
    # print(sequential_labels)
    # stop()
    return np.array(sequential_trials), np.array(sequential_labels), trial_dict, np.array(new_time_index)
                

def aggregate_event_times_matched(study, settings):

    # (cell, ses, events)
    agg_events = []
    # (cell, ses, cell_obj)
    agg_event_objects = []
    # (cell ,ses, binary events)
    agg_events_binary = []
    # (cell ,ses, trial ids/labels)
    agg_trial_types = []
    prev_time_index = None
    for animal in study.animals:

        max_matched_cell_count = len(animal.sessions[sorted(list(animal.sessions.keys()))[-1]].get_cell_data()['cell_ensemble'].cells)

        # for every cell in animal
        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1
            cell_events = []
            cell_event_objects = []
            cell_events_binary = []
            cell_trial_order = []

            # for every ses in animal that has that cell
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                ses = animal.sessions[seskey]
                ensemble = ses.get_cell_data()['cell_ensemble']
                if cell_label in ensemble.get_cell_label_dict():
                    cell = ensemble.get_cell_by_id(cell_label)
                    cell_events.append(cell.event_times)
                    cell_event_objects.append(cell)

                    # binary 
                    ct, time_index = event_times_to_count(cell.event_times, cell.cluster.time_index[-1], settings['bin_timestep'])
                    if prev_time_index is not None:
                        assert time_index.all() == prev_time_index.all()
                    cell_events_binary.append(ct)
                    prev_time_index = time_index
                    trial_order = list(cell.cluster.session_metadata.file_paths['tet'].split('-')[-1].split('.')[0].split('odor')[-1])
                    # print(trial_order)
                    if settings['firstIsBaseline']:
                        trial_order = np.hstack((['o'], trial_order))
                    # print(trial_order)
                    # cell_trial_order = np.hstack((cell_trial_order, trial_order))
                    cell_trial_order.append(trial_order)
            agg_events.append(cell_events)
            agg_event_objects.append(cell_event_objects)
            agg_events_binary.append(cell_events_binary)
            agg_trial_types.append(cell_trial_order)

    # agg_events saves as (cell, sessions_cell_is_in, event_times)
    # print(agg_trial_types)
    return np.array(agg_events), np.array(agg_event_objects), np.array(agg_events_binary), np.array(agg_trial_types)
