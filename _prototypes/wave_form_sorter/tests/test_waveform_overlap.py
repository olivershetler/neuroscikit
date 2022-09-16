import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.wave_form_sorter.waveform_overlap import waveform_overlap

def test_waveform_overlap():
    """
    Test waveform_overlap
    """
    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 0, 0)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 1

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (1, 1, 1)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 1, 1)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 1, 0)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 0, 1)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 1, 2)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 2, 1)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 2, 3)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5

    waveform_template_1 = (0, 0, 0)
    waveform_template_2 = (0, 1, 2)
    assert waveform_overlap(waveform_template_1, waveform_template_2) == 0.5