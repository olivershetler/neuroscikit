import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

@pytest.mark.skip(reason="Not implemented yet")
def test_unit_divergences():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_match_filter(): # pull out all the units whose min divergences match perfectly and are less than a threshold
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_match_units(): # Hungerian Algorithm
    pass