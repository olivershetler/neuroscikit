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

    for i in range(sample_length):
        time.append(i*dt)

    return time



class SpikeKeys():
    def __init__(self):
        self.sample_length = [
            'sample_length',
            'length',
            'len',
        ]
        self.sample_rate = [
            'sample_rate',
            'rate',
            'r',
        ]
        self.spikes_binary = [
            'spikes_binary',
            'binary',
            'bina',
        ]
        self.spike_times = [
            'spike_times',
            'times',
            'ts',
            't',
            'sptimes',
            'spks',
            'sps',
        ]
        self.cluster_labels = [
            'cluster_labels',
            'labels',
            'label',
            'cluster_label',
        ]

    def get_spike_train_init_keys(self):
        init_keys = [
            'sample_length', 
            'sample_rate', 
            'spikes_binary', 
            'spike_times',
        ]
        return init_keys


class SpikeTypes():
    def __init__(self):
        self.sample_length = int
        self.sample_rate = float
        self.spike_times = list[float]
        self.spikes_binary = list[int]
        self.cluster_labels = list[int]
        self.init_keys = []

    # def _set_init_keys(self, init_keys):
    #     self.init_keys = init_keys

    def type_dict(self):
        types = {
        'sample_length': int(0),
        'sample_rate': float(0.0),
        'spike_times': [],
        'spikes_binary': [],
        'cluster_labels': [],
        }
        return types

    def format_keys(self, init_keys):
        self.init_keys = init_keys

        input_dict = {}
        types = self.type_dict()

        assert len(self.init_keys) != 0, 'make sure to set initial dict. keys using SpikeKeys class'

        for i in range(len(self.init_keys)):
            if self.init_keys[i] in types:
                input_dict[str(self.init_keys[i])] = types[self.init_keys[i]]
        
        return input_dict

