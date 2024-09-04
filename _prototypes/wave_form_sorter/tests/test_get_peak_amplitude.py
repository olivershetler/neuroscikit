import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.wave_form_sorter.get_peak_amplitudes import get_peak_amplitudes


def make_waveforms(channel_count, spike_count, samples_per_wave):
    waveforms = np.zeros((channel_count, spike_count, samples_per_wave))

    for i in range(channel_count):
        for j in range(samples_per_wave):
            waveforms[i,:,j] = np.random.randint(-20,20,size=spike_count).tolist()

    return waveforms.tolist()


def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time



def test_get_peak_ampltiude():

    spike_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)


    peak, peak_ids = get_peak_amplitudes(waveforms)
    assert type(peak) == list
    assert type(peak_ids) == list
    assert type(peak[0]) == list
    assert type(peak_ids[0]) == list
    assert type(peak[0][0]) == list
    assert type(peak_ids[0][0]) == list
            





if __name__ == '__main__':
    test_get_peak_ampltiude()