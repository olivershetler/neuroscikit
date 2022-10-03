import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read
from _prototypes.unit_matcher.unit import (
    multivariate_kullback_leibler_divergence
)

@pytest.mark.skip(reason="Not implemented yet")
def test_multivariate_kullback_leibler_divergence():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_jensen_shannon_divergence():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_reduce_dimensionality():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def extract_unit_features():
    pass