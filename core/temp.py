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

        # channel_numbers = [i for i in range(1,9,1)]

        self.channel1 = [
            'channel1',
            'channel_1',
            'ch1',
            'chan1',
            'ch_1',
            'chan_1'
        ]

    def get_channels(self):
        # regex to match variations of ch1/2/3/4
        # return sorted channel list

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