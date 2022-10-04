import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

from _prototypes.unit_matcher.spike import waveform_level_features

spike = read.spike
delta = read.delta

def test_waveform_level_features():
    features = waveform_level_features(spike, delta)
    assert features is not None
    assert type(features) == dict
    for key, value in features.items():
        assert type(key) == str
        assert type(value) == float or type(value) == int
        assert value is not None
        assert value != float('inf')
        assert value != float('-inf')
        assert value != float('nan')

@pytest.mark.skip(reason="Not implemented yet")
def test_localize_source():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_extract_spike_features():
    pass