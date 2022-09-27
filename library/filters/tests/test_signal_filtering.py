import os
from random import sample
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

from core.ephys import (
    EphysCollection,
    EphysSeries,
)

from library.filters.__init__ import *

# from library.custom_cheby import custom_cheby1
# from library.dc_blocker_filter import dcblock
# from library.fast_fourier_transform import fast_fourier
# from library.infinite_impulse_response_filter import iirfilt
# from library.notch_filter import notch_filt


from x_io.rw.intan.load_intan_rhd_format.load_intan_rhd_format import read_rhd_data
from x_io.rw.intan.read_rhd import read_rhd

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
data_dir = os.path.join(parent_dir, 'neuroscikit_test_data')
test_rhd_file_path = os.path.join(data_dir, 'sampledata.rhd')

np.random.seed(0)

# Figure out inputs for test
def test_custom_cheby():
    collection = read_rhd(test_rhd_file_path)

    N = 100
    Rp = 10
    Wp = 20
    Ws = Wp * 2

    filtered = custom_cheby1(collection.signal['channel_0'], N, Rp, Wp, Ws)

    collection.signal['channel_0'].add_filtered(filtered, method='cheby1', type='butter')
    filtered_dict = collection.signal['channel_0'].get_filtered_dict()

    collection.signal['channel_0'].set_default_filter(method='cheby1', type='butter')

    assert filtered_dict['cheby1']['butter'].all() == filtered.all()
    assert collection.signal['channel_0'].filtered.all() == filtered.all()

# Figure out inputs for test
def test_dc_blocker_filter():
    collection = read_rhd(test_rhd_file_path)

    filtered = dcblock(collection.signal['channel_0'], fc=100)

    collection.signal['channel_0'].add_filtered(filtered, method='dcblock', type='butter')
    filtered_dict = collection.signal['channel_0'].get_filtered_dict()

    collection.signal['channel_0'].set_default_filter(method='dcblock', type='butter')

    assert filtered_dict['dcblock']['butter'].all() == filtered.all()
    assert collection.signal['channel_0'].filtered.all() == filtered.all()

def test_fast_fourier_transform():
    collection = read_rhd(test_rhd_file_path)
    data = collection.signal['channel_0'].signal

    frq, FFT_norm = fast_fourier(collection.signal['channel_0'])

    collection.signal['channel_0'].fft = frq
    collection.signal['channel_0'].fft_norm = FFT_norm

    assert len(frq) == len(data)/2
    assert len(FFT_norm) == len(data)/2

# Figure out inputs for test
def test_infinite_impulse_response_filter():
    collection = read_rhd(test_rhd_file_path)

    Wp = 10
    Ws = 20

    filtered = iirfilt(collection.signal['channel_0'], 'band', Wp, Ws=Ws, order=3, analog_val=False, automatic=0, Rp=3, As=60, filttype='butter')

    collection.signal['channel_0'].add_filtered(filtered, method='iirfilt', type='butter')
    filtered_dict = collection.signal['channel_0'].get_filtered_dict()

    collection.signal['channel_0'].set_default_filter(method='iirfilt', type='butter')

    assert filtered_dict['iirfilt']['butter'].all() == filtered.all()
    assert collection.signal['channel_0'].filtered.all() == filtered.all()

def test_notch_filter():
    collection = read_rhd(test_rhd_file_path)
    data = collection.signal['channel_0'].signal
    fs = collection.signal['channel_0'].sample_rate[0]

    filtered = notch_filt(collection.signal['channel_0'], band=10, freq=60, ripple=1, order=2, filter_type='butter', analog_filt=False)

    collection.signal['channel_0'].add_filtered(filtered, method='notch_filt', type='butter')
    filtered_dict = collection.signal['channel_0'].get_filtered_dict()

    collection.signal['channel_0'].set_default_filter(method='notch_filt', type='butter')

    assert filtered_dict['notch_filt']['butter'].all() == filtered.all()
    assert collection.signal['channel_0'].filtered.all() == filtered.all()


if __name__ == '__main__':
    pass
    # test_custom_cheby()
    # test_dc_blocker_filter()
    # test_fast_fourier_transform()
    # test_infinite_impulse_response_filter()
    # test_notch_filter()

