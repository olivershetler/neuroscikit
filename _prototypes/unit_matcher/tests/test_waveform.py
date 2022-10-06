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
    ,zero_crossings
    ,local_extrema
    ,area_under_curve
    ,symmetric_logarithm
    ,Point
    ,morphological_points
    ,waveform_features
)

from matplotlib.pyplot import plot, legend, show, axvline, axhline
import numpy as np

waveform = read.waveform
time_step = read.time_step

# Domain Conversion Functions

def test_time_index():
    assert len(waveform) == len(time_index(waveform, time_step))
    plot(time_index(waveform, time_step), waveform)

def test_derivative():
    assert len(waveform) == len(derivative(waveform, time_step))
    plot(time_index(waveform, time_step), derivative(waveform, time_step), linestyle='--')

def test_derivative2():
    assert len(waveform) == len(derivative2(waveform, time_step))
    #plot(time_index(waveform, time_step), derivative2(waveform, time_step), linestyle=':')


# Feature Utility Functions

@pytest.mark.skip(reason="Tested implicitly through morphological_points")
def test_local_extrema():
    d_waveform = derivative(waveform, time_step)
    extrema = local_extrema(waveform, time_step)
    d_extrema = local_extrema(d_waveform, time_step)
    for e in extrema:
        plot(time_index(waveform, time_step)[e], d_waveform[e], 'o')
    for e in d_extrema:
        axvline(time_index(waveform, time_step)[e], linestyle=':')
    show()

# Key morphological point objects

def test_point():
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)
    p = Point(10, t, waveform, d_waveform, d2_waveform)
    assert p.t == t[10]
    assert p.v == waveform[10]
    assert p.dv == d_waveform[10]
    assert p.d2v == d2_waveform[10]
    assert type(p) == Point
    assert type(p.i) == int
    assert type(p.t) == float
    assert type(p.v) == float
    assert type(p.dv) == float
    assert type(p.d2v) == float


def test_morphological_points():
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)

    p1, p2, p3, p4, p5, p6 = morphological_points(t, waveform, d_waveform, d2_waveform, time_step)

    # Plot the waveform and the morphological points
    plot(t, waveform)
    plot(t, d_waveform, linestyle='--')
    plot(p1.t, p1.v, 'o', label='p1')
    plot(p2.t, p2.dv, 'o', label='p2')
    plot(p3.t, p3.v, 'o', label='p3')
    plot(p4.t, p4.dv, 'o', label='p4')
    plot(p5.t, p5.v, 'o', label='p5')
    plot(p6.t, p6.dv, 'o', label='p6')
    legend()
    show()

# Main Feature Extraction Function

def test_waveform_features():
    i=-1
    for wf in read.session_dict1['ch4']:
        i+=1
        try:
            feature_vector = waveform_features(wf, time_step)
        except:
            print('Error in waveform', i)
            plot_waveform_points(wf, time_step)

def plot_waveform_points(waveform, time_step):
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)

    p1, p2, p3, p4, p5, p6 = morphological_points(t, waveform, d_waveform, d2_waveform, time_step)

    # Plot the waveform and the morphological points
    plot(t, waveform)
    plot(t, d_waveform, linestyle='--')
    plot(p1.t, p1.v, 'o', label='p1')
    plot(p2.t, p2.dv, 'o', label='p2')
    plot(p3.t, p3.v, 'o', label='p3')
    plot(p4.t, p4.dv, 'o', label='p4')
    plot(p5.t, p5.v, 'o', label='p5')
    plot(p6.t, p6.dv, 'o', label='p6')
    legend()
    show()