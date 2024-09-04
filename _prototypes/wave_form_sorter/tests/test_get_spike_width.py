import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.wave_form_sorter.spike_statistics import get_spike_statistics

def make_waveforms(channel_count, spike_count, samples_per_wave):
    waveforms = np.zeros((spike_count, channel_count, samples_per_wave))

    for i in range(channel_count):
        for j in range(samples_per_wave):
            waveforms[:,i,j] = np.random.randint(-20,20,size=spike_count).tolist()
            for k in range(10):
                waveforms[i,:,j] += np.random.rand()

    return waveforms.tolist()


def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time




def test_get_spike_statistics():
    spike_times = make_1D_timestamps()
    ch_count = 4
    samples_per_wave = 50
    waveforms = make_waveforms(ch_count, len(spike_times), samples_per_wave)

    waveform_dict = get_spike_statistics(np.array(waveforms), 50)

    assert type(waveform_dict) == dict
    assert 'pp_amp' in waveform_dict.keys()

if __name__ == '__main__':
    test_get_spike_statistics()