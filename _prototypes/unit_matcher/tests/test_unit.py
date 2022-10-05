import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import numpy as np
import _prototypes.unit_matcher.tests.read as read
from _prototypes.unit_matcher.unit import (
    multivariate_kullback_leibler_divergence
    ,jensen_shannon_distance
)

P = np.random.rand(150, 24)
Q = np.random.rand(190, 24)

def test_multivariate_kullback_leibler_divergence():
    print(multivariate_kullback_leibler_divergence(P, Q))
    print(multivariate_kullback_leibler_divergence(Q, P))
    #print(multivariate_kullback_leibler_divergence(P, P))


def test_jensen_shannon_divergence():
    print(jensen_shannon_distance(P, Q))
    print(jensen_shannon_distance(Q, P))
    #assert jensen_shannon_distance(P, Q) == jensen_shannon_distance(Q, P)

@pytest.mark.skip(reason="Not implemented yet")
def test_reduce_dimensionality():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def extract_unit_features():
    pass