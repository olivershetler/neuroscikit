from _prototypes.waveform_overlap import waveform_overlap

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