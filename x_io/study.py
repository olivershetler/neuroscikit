from abc import abstractmethod
from enum import unique
import os
from re import S
import sys
import wave

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

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

from datetime import datetime, timedelta
from random import sample

def make_hms_index_from_rate(start_time, sample_length, sample_rate):
    """
    Creates a time index for a sample of length sample_length at sample_rate
    starting at start_time.

    Output is in hours, minutes, seconds (HMS)
    """
    if type(start_time) == str:
        start_time = datetime.strptime(start_time, '%H:%M:%S')
    time_index = [start_time]
    for i in range(1,sample_length):
        time_index.append(time_index[-1] + timedelta(seconds=1/sample_rate))
    str_time_index = [time.strftime('%H:%M:%S.%f') for time in time_index]
    return str_time_index

def make_seconds_index_from_rate(sample_length, sample_rate):
    """
    Same as above but output is in seconds, start_time is automatically 0
    Can think of this as doing all times - start_time so we have 0,0.02,0.04... array etc..
    """
    start_time = 0
    dt = 1/sample_rate

    time = []

    for i in range(start_time, int(sample_length*sample_rate)):
        time.append(i*dt)

    return time





