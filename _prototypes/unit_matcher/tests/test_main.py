import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

@pytest.mark.skip(reason="Not implemented yet")
def test_read():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_write():
    # read a file
    # use a fixed re-mapping scheme
    # write a new cut file
    # read the new cut file
    # compare the new cut file to the original
    # assert that the new cut file conforms to the mapping scheme
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_main():
    pass