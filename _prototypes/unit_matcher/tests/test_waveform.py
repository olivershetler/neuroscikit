import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

from _prototypes.unit_matcher.waveform import (
    time_index
    ,derivative
    ,derivative2
    ,area_under_curve
)

waveform = read.waveform
delta = read.delta

# Domain Conversion Functions

def test_time_index():
    assert len(waveform) == len(time_index(waveform, delta))

def test_derivative():
    assert len(waveform) == len(derivative(waveform, delta))

def test_derivative2():
    assert len(waveform) == len(derivative2(waveform, delta))


# Feature Utility Functions

@pytest.mark.skip(reason="Not sure how to test this one.")
def test_area_under_curve():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_maximum_positive_peak():
    pass

# Key morphological point objects



# Main Feature Extraction Function

@pytest.mark.skip(reason="Not implemented yet")
def test_extract_waveform_features():
    pass