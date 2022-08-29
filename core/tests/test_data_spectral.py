import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from core.data_spectral import (
    SpectralSeries
)

def make_input_dict():
    input_dict = {
        'sample_rate': 50,
        'sample_length': 2,
    }

    return input_dict

def test_spectral_series():
    input_dict = make_input_dict()

    spectral_series = SpectralSeries(input_dict)

    assert spectral_series.sample_rate == input_dict['sample_rate']
    assert spectral_series.sample_length == input_dict['sample_length']
    assert type(spectral_series.get_input_dict()) == dict
    assert spectral_series.get_input_dict() == input_dict
    assert type(spectral_series.get_filtered()) == list
    assert type(spectral_series.power_bands) == list

if __name__ == '__main__':
    test_spectral_series