from abc import abstractmethod
from enum import unique
import os
from re import S
import sys
import wave

from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import (
    make_seconds_index_from_rate,
)

class Study():
    """
    Top level class, holds all study information with data for each animal
    """
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict
        self.sample_length, self.sample_rate, self.animal_ids = self._read_input_dict()

        self.timebase = make_seconds_index_from_rate(self.sample_length, self.sample_rate)
        self.animals = []
        self.agg_stat_dict = None

    def _read_input_dict(self):
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']
        animal_ids = self._input_dict['animal_ids']
        ## Add init input info for study
        return sample_length, sample_rate, animal_ids

    def add_animal(self, subj_dict: dict):
        subj_dict['timebase'] = self.timebase
        self.animals.append(Animal(subj_dict))
        self.animal_ids.append(subj_dict['id'])

    def get_pop_spike_times(self):
        spike_times = []
        for i in range(len(self.animals)):
            spike_times.append(self.animals[i].agg_spike_times)
        return spike_times

    def get_animal_ids(self):
        return self.animal_ids

    def get_animal(self, id):
        return self.animals[id] 

    def get_animal_stats(self):
        if self.agg_stat_dict == None:
            for id in self.animal_ids:
                self.agg_stat_dict[id] = self.get_animal(id).stat_dict
        return self.agg_stat_dict


# Dictionary:
# Animals

class Animal():
    """
    Holds all session information and data for an animal

    Input dict format: 'timebase', 0, 1, 2, etc..
    """
    def __init__(self, input_dict: dict):
        self.sessions = input_dict
        self.timebase, self.agg_spike_times, self.agg_cluster_labels, self.agg_events, self.agg_waveforms, self.session_count, self.id, self.session_keys = self._read_input_dict() 
        self.stat_dict = None
        self.spatial_dict = None
        self.agg_sorted_events = None
        self.agg_sorted_waveforms = None
        self.agg_cell_keys = None
        self.agg_sorted_labels = None

    def add_sorted_data(self, good_events, good_waveforms, good_labels):
        self.agg_sorted_events = good_events
        self.agg_sorted_waveforms = good_waveforms
        self.agg_sorted_labels = good_labels
        self.agg_cell_keys = []
        for ses in good_events:
            self.agg_cell_keys.append([i for i in range(0, len(ses))])

    def _read_input_dict(self):
        agg_spike_times = []
        agg_cluster_labels = []
        agg_waveforms = []
        agg_events = []
        session_keys = []
        count = 0
        for session in self.sessions:
            if session == 'timebase':
                timebase = self.sessions[session]
            elif session == 'id':
                animal_id = self.sessions[session]
            else:
                session_keys.append(session)
                count += 1
                cluster_labels = self.sessions[session]['cluster_labels']
                spike_times = self.sessions[session]['spike_times']
                waveforms = self._extract_waveforms(self.sessions[session])
                events = self._fill_events(spike_times, cluster_labels, waveforms)
                assert type(spike_times) == list, 'Spike times are not a list, check inputs'
                agg_spike_times.append(spike_times)
                agg_cluster_labels.append(cluster_labels)
                agg_waveforms.append(waveforms)
                agg_events.append(events)
        return timebase, agg_spike_times, agg_cluster_labels, agg_events, agg_waveforms, count, animal_id, session_keys

    def _fill_events(self, spike_times, cluster_labels, waveforms):
        events = []
        for i in range(len(spike_times)):
            event_waves = []
            for j in range(len(waveforms)):
                event_waves.append(waveforms[j][i])
            events.append(Spike(spike_times[i], cluster_labels[i], event_waves))
        return events

    def _extract_waveforms(self, session):
        waveforms = []
        ch_keys = sorted([x for x in session.keys() if 'ch' in x])
        for channel in ch_keys:
            waveforms.append(session[channel])
        return waveforms

    def add_session(self, session_dict):
        cluster_labels = session_dict['cluster_labels']
        spike_times = session_dict['spike_times']
        assert type(spike_times) == list, 'Spike times are not a list, check inputs'

        waveforms = self._extract_waveforms(session_dict)
        events = self._fill_events(spike_times, cluster_labels, waveforms)

        keys = list(self.sessions.keys())
        keys = [int(x) for x in keys if str(x).isnumeric()]

        # self.sessions[str(max(keys) + 1)] = session_dict
        self.sessions[int(max(keys) + 1)] = session_dict
        self.agg_spike_times.append(spike_times)
        self.agg_cluster_labels.append(cluster_labels)
        self.agg_waveforms.append(waveforms)
        self.agg_events.append(events)
        self.session_count += 1

    def get_session_data(self, id):
        return self.sessions[int(id)]

    def get_stat_dict(self):
        assert self.agg_cell_keys != None, 'Need to sort your spike times by cell, use "sort_cell_spike_times" in library'
        if self.stat_dict == None:
            self.stat_dict = {}
            self.stat_dict['session_stats'] = {} # --> session 1, session 2, etc.. --> session stats
            self.stat_dict['animal_stats'] = {} # --> anmal stats
            self.stat_dict['cell_stats'] = {} # session 1, session 2, ... --> cell 1, cell 2, ... --> cell stats
            c = 0
            for session in self.session_keys: 
                self.stat_dict['session_stats'][session] = {}
                self.stat_dict['cell_stats'][session] = {}
                for cell in self.agg_cell_keys[c]:
                    self.stat_dict['cell_stats'][session][cell] = {}
                c += 1
        return self.stat_dict

    def get_spatial_dict(self):
        if self.spatial_dict == None:
            self.spatial_dict = {}
            for session in self.session_keys:
                self.spatial_dict[session] = {}
        return self.spatial_dict

    def clear_stat_dict(self):
        self.stat_dict = {}
        self.stat_dict['session_stats'] = {} # --> session 1, session 2, etc.. --> session stats
        self.stat_dict['animal_stats'] = {} # --> anmal stats
        self.stats_dict['cell_stats'] = {} # session 1, session 2, ... --> cell 1, cell 2, ... --> cell stats
        for session in self.sessions: 
            self.stat_dict['session_stats'][session] = {}

    def add_single_cell_stat(self, session, cell, cell_stats):
        self.get_stat_dict()
        self.stat_dict['cell_stats'][session][cell] = cell_stats

    def add_cell_stat(self, sessions, cells, cell_stats):
        self.get_stat_dict()
        for i in range(len(sessions)):
            for cell in cells:
                self.stat_dict['cell_stats'][sessions[i]][cell] = cell_stats[sessions[i]][cell]

    def add_session_stat(self, sessions, session_stats):
        self.get_stat_dict()
        for i in range(len(sessions)):
            self.stats_dict['session_stats'][sessions[i]] = session_stats[sessions[i]]

    def add_spatial_stat(self, sessions, spatial_data: dict):
        """
        Spatial data is dictionary that holds spatial info to add to sesion e.g. pos_x: x position of arena
        """
        self.get_spatial_dict()
        for session in sessions:
            self.spatial_dict[session] = spatial_data[session]

    # def add_session_stat(self, session, statkey, statval, multiSession=False, multiStats=False):
    #     self.stat_dict = self.get_stat_dict()
    #     if multiSession == False and multiStats == False:
    #         self.stat_dict['session_stats'][session][statkey] = statval 
    #     elif multiSession == True and multiStats == False:
    #         assert type(session) == list, 'multiSession is true but only single session id provided'
    #         for i in range(len(session)):
    #             self.stat_dict['session_stats'][session[i]][statkey] = statval 
    #     elif multiSession == True and multiStats == True:
    #         assert len(session) == len(statkey), 'multiSEssion and multiStats true so session, statkey and stataval need to be lists of same len'
    #         assert len(statkey) == len(statval)
    #         for i in range(len(session)):
    #             self.stat_dict['session_stats'][session[i]][statkey[i]] = statval[i]
    #     elif multiSession == False and multiStats == True: 
    #         assert type(statkey) == list, 'multiStats is true but only single stat key provided'
    #         assert len(statkey) == len(statval), 'multiStats is true but only single stat val provided'
    #         for i in range(len(statkey)):
    #             self.stat_dict['session_stats'][session][statkey[i]] = statval[i]
        
    def add_animal_stat(self, animal_stats):
        self.stat_dict = self.get_stat_dict()
        self.stat_dict['animal_stats'] = animal_stats
    



class Event():
    def __init__(self, event_time, event_label, event_signal):
        self.event_time = event_time
        self.event_label = event_label
        self.event_signal = event_signal

        # check if signal is 2D or 1D (e.g. multiple channel waveforms or single channel signal)
        if type(event_signal[0]) == list:
            self.main_ind = 0
            self.main_signal = 0
        else:
            self.main_ind = None
            self.main_signal = None
 
    def set_label(self, label):
        self.event_label = label

    def get_signal(self, ind):
        return self.waveforms[ind]
    
    def get_peak_signal(self):
        if self.main_ind == 0:
            self.main_ind, self.main_signal = self._set_peak()
            return self.main_ind, self.main_signal
        else:
            print('event signal is 1 dimensional')
            return self.main_ind, self.main_signal

    def _set_peak(self):
        curr = 0
        for i in range(len(self.event_signal)):
            if max(self.event_signal[i]) > curr:
                curr = i + 1
        assert curr != 0, 'There is no 0 channel, make sure max(abs(channel waveform)) is not 0'
        return curr, self.event_signal[curr-1]

    

class Spike(Event):
    def __init__(self, spike_time: float, cluster_label: int, waveforms: list):
        super().__init__(spike_time, cluster_label, waveforms)
        self.cluster = cluster_label
        self.waveforms = waveforms
        self.spike_time = spike_time

        assert type(cluster_label) == int, 'Cluster label must be integer for index into waveforms'
        assert type(spike_time) == float, 'Spike times is in format: ' + str(type(spike_time))

        # self.main_ind = 0
        # self.main_signal = 0

    # one waveform per channel bcs class is for one spike
    # def get_single_channel_waveform(self, id):
    #     assert id in [1,2,3,4,5,6,7,8], 'Channel number must be from 1 to 8'
    #     return self.waveforms[id-1]
    
    # # get waveform with largest positive or negative deflection (peak or trough, absolute val)
    # def get_peak_channel(self):
    #     if self._main_channel == 0:
    #         self._main_channel, self._main_waveform = self._set_peak_channel()
    #         return self._main_channel, self._main_waveform
    #     else:
    #         return self._main_channel, self._main_waveform

    # # lazy eval, called when get main channel called
    # def _set_peak_channel(self):
    #     curr = 0
    #     for i in range(len(self.waveforms)):
    #         if max(self.waveforms[i]) > curr:
    #             curr = i + 1
    #     assert curr != 0, 'There is no 0 channel, make sure max(abs(channel waveform)) is not 0'
    #     return curr, self.waveforms[curr-1]

