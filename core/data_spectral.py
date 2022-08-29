import os
from re import S
import sys
import wave

"""
This module is for spectral data extracted from voltage data.
It corresponds to the frequency domain of the voltage data.
"""

class SpectralSeries():
    def __init__(self, input_dict: dict):

        self._input_dict = input_dict

        sample_length, sample_rate = self._read_input_dict()

        self.power_bands = []
        self.data = {}
        self._filtered = []
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        
    def _read_input_dict(self):
        sample_length = self._input_dict['sample_length']
        sample_rate = self._input_dict['sample_rate']

        return sample_length, sample_rate, 

    def get_input_dict(self):
        return self._input_dict

    def _format_data_to_filter(self):
        # not sure which aspects of input dict will be needed for which filtering fxns
        # this fxn will deal with any formatting necessary to output data dictionary for filtering
        # self.data = {
        #     'fs': self.sample_rate,
        #     'data': ,
        # }
        # return self.data
        pass

    def get_data_to_filter(self):
        if len(self.data.keys()) == 0:
            self._format_data_to_filter()
        return self.data

    def get_filtered(self):
        return self._filtered

    def _set_filtered(self, filtered):
        self._filtered = filtered

