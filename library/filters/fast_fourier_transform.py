import numpy as np
from scipy import fftpack

def fast_fourier(Fs, Y):
    '''Takes the Sample Frequency: Fs(Hz), the numer of samples, N, and the data values (Y),
    and performs a Fast Fourier Transformation to observe the signal in the frequency domain'''

    N = len(Y)
    k = np.arange(N)

    T = N / Fs

    frq = k / T  # two sides frequency range
    frq = frq[range(int(N / 2))]  # one side frequency range

    FFT = fftpack.fft(Y)  # fft computing and normalization

    FFT_norm = 2.0 / N * np.abs(FFT[0:int(N / 2)])

    return frq, FFT_norm