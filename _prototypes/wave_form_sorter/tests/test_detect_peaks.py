import os
import sys
import wave
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.wave_form_sorter.detect_peaks import detect_peaks


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



def test_detect_peaks():

    spike_times = make_1D_timestamps()
    ch_count = 8
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)

    for j in range(len(waveforms)):
        for i in range(len(waveforms[j])):
            ind = detect_peaks(waveforms[j][i])
            assert type(ind) == np.ndarray




if __name__ == '__main__':
    test_detect_peaks()