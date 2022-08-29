"""
This module is for local field potential (LFP) data. These data are measured
"""

class EphysSeries():
    """
    This class is for local field potential (LFP) data. These data are measured
    """
    def __init__(self, data_dict: dict):
        """
        Initialize the LFP object.
        """
        if 'ephys_series' in data_dict:
            self.data = data_dict['ephys_series']
        if 'units' in data_dict:
            self.units = data_dict['units']
        if 'sample_rate' in data_dict:
            self.sample_rate = data_dict['sample_rate']

    def check_data_types(data_dict: dict):
        """
        Check the data types of the data dictionary.
        """
        if 'ephys_series' in data_dict:
            series_type = type(data_dict['ephys_series'])
            assert series_type == list or series_type == tuple, 'The ephys_series must be a list or tuple. The type of the ephys_series you tried to add is {}'.format(series_type)
            assert type(data_dict['ephys_series'][0]) == int or type(data_dict['ephys_series'][0]) == float, 'The data type of the ephys_series must be int or float. The data type of the firse element in the ephys_series is {}'.format(type(data_dict['ephys_series'][0]))
        if 'units' in data_dict:
            units_type = type(data_dict['units'])
            assert units_type == str, 'The units must be a string. The type of the units you tried to add is {}'.format(units_type)
        if 'sample_rate' in data_dict:
            sample_rate_type = type(data_dict['sample_rate'])
            assert sample_rate_type == dict or sample_rate_type == tuple or sample_rate_type == list, 'The sample_rate must be an int or float. The type of the sample_rate you tried to add is {}'.format(sample_rate_type)
            assert len(data_dict['sample_rate']) == 2, 'The sample_rate must be a tuple or list of length 2. The length of the sample_rate you tried to add is {}'.format(len(data_dict['sample_rate']))
            assert type(data_dict['sample_rate'][0]) == int or type(data_dict['sample_rate'][0]) == float, 'The data type of the sample_rate must be int or float. The data type of the firse element in the sample_rate is {}'.format(type(data_dict['sample_rate'][0]))
            assert type(data_dict['sample_rate'][1]) == str, 'The data type of the sample_rate must be a string. The data type of the second element in the sample_rate is {}'.format(type(data_dict['sample_rate'][1]))

class EphysCollection():
    def __init__(self, channel_dict: dict):
        """
        Initialize the LFP object.
        """
        #check_data_types(channel_dict)
        self.channels = channel_dict

    @staticmethod
    def check_data_types(channel_dict: dict):
        """
        Check the data types of the data dictionary.
        """
        for channel, ephys_series in channel_dict.items():
            assert type(ephys_series) == EphysSeries, 'The data type of the ephys_series must be EphysSeries. The data type of the ephys_series you tried to add is {}'.format(type(ephys_series))

